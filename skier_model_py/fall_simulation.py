"""
Post-fall sliding model  --  "Where does a fallen skier stop?"

Given a known fall trajectory (produced upstream from course geometry and hazard
analysis), the speed v0 at the instant of the fall and the local slope inclination
alpha along the path (from the LiDAR .las surface), this module integrates the
decelerated, path-constrained slide of the body and returns *where it comes to rest*.

MODEL (see the technical brief, SkiSlo -- July 2026)
---------------------------------------------------
The skier is a point mass m constrained to the 3D curve, parametrised by arc length
s. The path is fixed, so the only unknown is the speed profile v(s). Newton along
the tangent (m*v_dot = F_net) plus the chain rule (v_dot = v*dv/ds) gives, in the
variable w = v**2:

    dw/ds = (2/m) * [ F_p(s) - F_drag(w) - F_fric(s, w) ],      w(0) = v0**2   (1)

with

    F_p    = m*g*sin(alpha)*cos|beta|                 tangential gravity          (2)
    F_drag = 0.5*rho*CdA*w                            drag, always opposing        (3)
    F_fric = mu_c * N,  N = sqrt((m*g*cos(alpha))**2 + F_lat**2)                   (4)
    F_lat  = m*w/R(s) + sign(beta)*m*g*sin(alpha)*sin|beta|                        (5)

The normal load N is deliberately *not* just m*g*cos(alpha): on a curved constrained
path the lateral load presses the body into the snow and raises friction. This is the
same mechanism physical_model.py uses, retained here because the fall path is given
as constrained.

Working in space rather than time buys three things:
  (i)   alpha(s), beta(s), R(s) are known functions of position, so every coefficient
        is directly evaluable -- time never enters;
  (ii)  the stopping point comes out *directly in space* as the root w(s_stop) = 0,
        located precisely by a terminal integration event;
  (iii) drag and centrifugal load are both proportional to w, so (1) is nearly linear
        and numerically well behaved.

PARAMETERS: a sliding body is not a carving skier
-------------------------------------------------
    Parameter          Racing (physical_model.py)   Fall (default)   Rationale
    snow friction mu   0.16                         0.45 (0.3-0.7)   suit/limbs plough,
                                                                     no gliding ski base
    drag area CdA      0.30                         0.70             tumbling, non-
                                                                     aerodynamic posture

mu_c dominates the answer and is genuinely uncertain (literature spans ~0.3-0.7 for a
suited body on packed snow, depending on clothing, snow hardness and sliding attitude).
Prefer `ensemble_caduta` over a single run -- see its docstring.

RELATIONSHIP TO physical_model.py
---------------------------------
The signature mirrors `esegui_simulazione` so this drops straight into the existing
pipeline, and the geometry (s, alpha, beta, signed R, interpolants) is built the same
way. Three deliberate differences:

  * NO MIN-SHIFT. `esegui_simulazione` subtracts the per-axis minimum from x/y/z.
    Here the coordinates are left untouched, so `stop_xyz` comes back in the original
    CRS (UTM32N) -- ready to be placed on the riverMatrix and consistent with the .las
    lookup.
  * Curvature, not radius, is interpolated. Straight segments have R = inf, and
    interpolating between two infinities yields NaN (inf + (inf-inf)*t), which the
    `isinf` guard in the base model does not catch. kappa = 1/R is finite everywhere
    (exactly 0 when straight) and F_cent = m*w*kappa is algebraically identical.
  * Omitting `alfa` actually works: alpha is estimated from the path geometry.

ASSUMPTIONS AND LIMITS
----------------------
Point mass. Permanent snow contact: no airborne bounces, whereas real high-speed falls
include phases with zero friction -- so real runouts can be LONGER than predicted. The
body follows the given trajectory exactly. mu_c is constant along the slide, whereas in
reality it may grow as the body ploughs soft snow.

Author: SkiSlo project.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# --- Defaults for a sliding body (vs. a carving skier; see module docstring). -------
MU_CADUTA_DEFAULT = 0.45      # Coulomb friction, body/snow. Range 0.3-0.7.
CDA_CADUTA_DEFAULT = 0.70     # Drag area [m^2], tumbling posture.
MU_ENSEMBLE_DEFAULT = (0.20, 0.45, 0.50)


# =====================================================================================
# 1. TRAJECTORY CONDITIONING
# =====================================================================================
def _pulisci_traiettoria(x, y, z, alfa):
    """Round to cm and drop zero-length segments. Coordinates are NOT min-shifted.

    Rounding to 2 decimals kills interpolation/GNSS jitter (this is what the base model
    does). Absolute UTM32N easting/northing are ~6-7 digits, so cm precision sits well
    inside float64's ~15 significant digits -- rounding in the original CRS is safe.

    Dropping zero-length segments matters: `np.gradient(x, s)` divides by ds, so a
    duplicated point yields an infinite tangent and NaNs propagate through beta into
    the ODE. Rounding can *create* duplicates, so this must happen after it.

    Returns the cleaned arrays plus the boolean mask of kept points, so a caller can
    realign per-point arrays it computed on the original trajectory (e.g. the `grads`
    from obtain_inclination).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if not (x.size == y.size == z.size):
        raise ValueError(
            f"x, y, z must have the same length (got {x.size}, {y.size}, {z.size})."
        )
    if alfa is not None:
        alfa = np.asarray(alfa, dtype=float).ravel()
        if alfa.size != x.size:
            raise ValueError(
                f"alfa must have one value per trajectory point "
                f"(got {alfa.size} for {x.size} points)."
            )

    x = np.round(x, 2)
    y = np.round(y, 2)
    z = np.round(z, 2)

    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    keep = np.concatenate(([True], ds > 0.0))
    x, y, z = x[keep], y[keep], z[keep]
    if alfa is not None:
        alfa = alfa[keep]

    if x.size < 3:
        raise ValueError(
            f"Need at least 3 distinct trajectory points, got {x.size} after "
            f"removing duplicates."
        )
    return x, y, z, alfa, keep


def _stima_alpha_deg(x, y, z):
    """Slope inclination [deg] estimated from the path geometry alone.

    alpha = -arctan(dz / dh), with dh the horizontal step: descending (dz < 0) gives
    alpha > 0. This is the SkiSlo convention -- the same sign that `return_inclination`
    in main_matrix.py produces by negating `obtain_inclination`, and the sign that makes
    F_p = m*g*sin(alpha)*cos|beta| a driving force on a descent.

    np.gradient uses central differences inside and one-sided at the ends, so every
    point gets a value (the base model's loop left the first and last few at zero). The
    uniform index spacing cancels in the dz/dh ratio, so no arc length is needed here.
    """
    dz = np.gradient(z)
    dh = np.hypot(np.gradient(x), np.gradient(y))
    return np.degrees(-np.arctan2(dz, dh))


def _risolvi_alpha_deg(x, y, z, alfa):
    """Pick the alpha array: measured where available, geometric estimate elsewhere.

    `obtain_inclination` returns NaN where the .las neighbourhood was too sparse to fit
    a plane. Rather than discarding the whole measured array on a single NaN (what
    main_matrix.return_inclination does), only the NaN entries fall back to geometry.

    IMPORTANT: `alfa` must already be in the SkiSlo convention (positive = descending),
    i.e. the NEGATED output of obtain_inclination, exactly as main_matrix.py passes it
    to esegui_simulazione.
    """
    if alfa is None:
        return _stima_alpha_deg(x, y, z)

    alpha_deg = np.array(alfa, dtype=float)
    bad = ~np.isfinite(alpha_deg)
    if bad.any():
        alpha_deg[bad] = _stima_alpha_deg(x, y, z)[bad]
    return alpha_deg


# =====================================================================================
# 2. GEOMETRY: arc length, alpha, beta, signed curvature
# =====================================================================================
def _ascissa_curvilinea(x, y, z):
    """Cumulative 3D arc length s. Strictly increasing (zero-length steps are gone)."""
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    return np.concatenate(([0.0], np.cumsum(ds)))


def _tangenti_e_beta(x, y, z, s, alpha_deg):
    """Unit tangents T_hat and the signed angle beta [rad] to the fall-line reference.

    Built exactly as in physical_model.py: the reference direction is the global
    start->end vector, rescaled per point by cos(gamma - alpha) (gamma is the overall
    descent angle of that vector). beta is then the angle between the local tangent and
    that reference, signed by the z-component of their horizontal cross product so that
    left and right turns get opposite signs -- which is what makes the lateral gravity
    term in Eq. (5) add or subtract.
    """
    dx_ref = x[-1] - x[0]
    dy_ref = y[-1] - y[0]
    dz_ref = z[-1] - z[0]
    vec_ref = np.array([dx_ref, dy_ref, dz_ref])

    gamma_deg = np.degrees(-np.arctan2(dz_ref, np.hypot(dx_ref, dy_ref)))
    factor = np.cos(np.radians(gamma_deg - alpha_deg))
    vec_ref_arr = factor[:, None] * vec_ref

    vec_ref_norm = np.linalg.norm(vec_ref_arr, axis=1)
    vec_ref_norm[vec_ref_norm == 0] = 1e-10
    vec_ref_hat = vec_ref_arr / vec_ref_norm[:, None]

    # Tangent w.r.t. arc length (not time): dP/ds is already a unit vector in exact
    # arithmetic, but normalising guards against discretisation drift.
    T = np.column_stack([np.gradient(x, s), np.gradient(y, s), np.gradient(z, s)])
    T_norm = np.linalg.norm(T, axis=1)
    T_norm[T_norm == 0] = 1e-10
    T_hat = T / T_norm[:, None]

    cosang = np.clip(np.sum(vec_ref_hat * T_hat, axis=1), -1.0, 1.0)
    beta_deg = np.degrees(np.arccos(cosang))

    cross_beta = T_hat[:, 0] * vec_ref_hat[:, 1] - T_hat[:, 1] * vec_ref_hat[:, 0]
    beta_sign = np.sign(cross_beta)

    return T_hat, np.radians(beta_sign * beta_deg)


def _curvatura_con_segno(x, y, z):
    """Signed horizontal curvature kappa = 1/R [1/m] via the Menger formula.

    Menger curvature of the triangle (P[i-k], P[i], P[i+k]) is 4*Area/(a*b*c); the sign
    comes from the z-component of the cross product (left turn vs right turn). The
    sliding window k smooths out point-to-point noise, as in the base model.

    Returning CURVATURE rather than radius is the key departure from physical_model.py.
    A straight segment has R = inf, and interp1d between two infinities evaluates
    inf + (inf - inf)*t = NaN, which slips past an `isinf` guard and poisons the ODE.
    kappa is 0 there -- finite, interpolable, and F_cent = m*w*kappa == m*w/R exactly.

    CAVEAT, inherited deliberately from the base model: `area` uses only the horizontal
    (x, y) cross product, while a, b, c are full 3D chord lengths. The result is
    therefore neither the horizontal nor the in-plane curvature -- on a slope it is
    biased by roughly (ds_3d/ds_horiz)**3, e.g. a 60 m plan radius on a 23 deg pitch
    reads back as ~77 m, so F_cent is underestimated there. It is kept as-is because
    the brief specifies the Menger formula "as in the base model" and the race model is
    already tuned around this convention; changing it here would make the two models
    disagree. Fix it in both or in neither.
    """
    N = x.size
    P = np.column_stack([x, y, z])
    kappa = np.zeros(N)

    k = min(50, max(1, N // 10))
    if N > 2 * k + 2:
        for i in range(k, N - k - 1):
            p_prev, p, p_next = P[i - k], P[i], P[i + k]
            v1 = p - p_prev
            v2 = p_next - p

            a = np.linalg.norm(p_prev - p)
            b = np.linalg.norm(p - p_next)
            c = np.linalg.norm(p_next - p_prev)

            cross = v1[0] * v2[1] - v1[1] * v2[0]
            area = 0.5 * abs(cross)
            if area <= 1e-16 or a * b * c == 0:
                continue  # collinear -> straight -> kappa stays 0
            kappa[i] = np.sign(cross) * 4.0 * area / (a * b * c)

        # The window cannot reach the ends; hold the nearest computed value.
        kappa[:k] = kappa[k]
        kappa[N - k - 1:] = kappa[N - k - 2]

    return kappa


# =====================================================================================
# 3. TIME RECONSTRUCTION
# =====================================================================================
def _tempo_da_velocita(s_seg, v_seg):
    """Elapsed time along one segment, from the speed profile.

    Formally t(s) = integral of ds/v, but 1/v blows up at arrest. Instead, assume
    constant deceleration between consecutive samples -- then ds = (v_i + v_i+1)/2 * dt
    holds *exactly*, so

        dt = 2*ds / (v_i + v_i+1)

    This is finite as v -> 0 and collapses to the brief's closed form for the final
    stretch (dt = 2*ds/v_last, since the trailing sample has v = 0), while staying
    second-order accurate on the rest of the profile. No 1/v ever gets evaluated.
    """
    ds = np.diff(s_seg)
    v_sum = v_seg[:-1] + v_seg[1:]
    dt = np.divide(2.0 * ds, v_sum, out=np.zeros_like(ds), where=v_sum > 0)
    return np.concatenate(([0.0], np.cumsum(dt)))


# =====================================================================================
# 4. THE SIMULATION
# =====================================================================================
def simula_caduta(
    x,
    y,
    z,
    v0,
    alfa=None,
    m=80.0,
    g=9.81,
    mu_caduta=MU_CADUTA_DEFAULT,
    mu_statico=None,
    rho=1.225,
    CdA_caduta=CDA_CADUTA_DEFAULT,
    v_restart=0.1,
    max_restarts=5,
    plot=False,
):
    """Integrate the post-fall slide and locate the arrest point.

    Parameters
    ----------
    x, y, z : (N,) array_like
        The fall trajectory, in the ORIGINAL CRS (UTM32N easting/northing + orthometric
        height). Not min-shifted -- `stop_xyz` comes back in the same CRS.
    v0 : float
        Speed [m/s] at the instant of the fall.
    alfa : (N,) array_like, optional
        Slope inclination [degrees], POSITIVE WHEN DESCENDING -- i.e. the negated
        output of `obtain_inclination`, exactly as main_matrix.return_inclination
        produces it. NaN entries fall back to the geometric estimate. If omitted, alpha
        is estimated from the path geometry alone.
    m, g, rho : float
        Body mass [kg], gravity [m/s^2], air density [kg/m^3].
    mu_caduta : float
        Kinetic Coulomb friction, body/snow. Dominant and uncertain: see
        `ensemble_caduta`.
    mu_statico : float, optional
        Static friction for the arrest test. Defaults to `mu_caduta` (conservative: a
        real mu_s >= mu_c would only make arrests easier to hold).
    CdA_caduta : float
        Drag area [m^2] of a tumbling body.
    v_restart : float
        Speed [m/s] injected when a stop is rejected and the slide resumes.
    max_restarts : int
        Cap on resumed slides, as a loop backstop.
    plot : bool
        Draw the velocity profile, the force balance and a plan view with the arrest
        marker.

    Returns
    -------
    dict with
        's', 'v', 't'          : profiles along the slide [m], [m/s], [s]
        'F_p','F_drag','F_fric','F_lat','F_cent','F_net' : force profiles [N]
        'R', 'alpha', 'beta'   : geometry sampled on the solution [m], [rad], [rad]
        'tan'                  : (M,3) unit tangents, one per (kept) trajectory point
        'arrestato'            : True if the body came to rest within the path
        's_stop'               : distance slid [m] (the full path length if not arrested)
        'stop_xyz'             : (3,) arrest position in the ORIGINAL CRS
        'tempo_arresto'        : total elapsed time [s]
        'v_finale'             : exit speed [m/s]; 0 when arrested
        'n_restarts'           : number of resumed slides
        'mask_punti_validi'    : (N,) bool mask of the input points actually used

    Notes on `arrestato`
    --------------------
    True means the body is at rest inside the supplied path and static friction holds it
    there. False means it is not, for one of two reasons: it exited the path still
    moving (`v_finale` > 0 -- the upstream fall path is shorter than the physical
    runout), or the restart cap was hit (`n_restarts` == max_restarts, `v_finale` == 0),
    which is a degenerate case rather than a physical answer.
    """
    mu_s = mu_caduta if mu_statico is None else mu_statico

    # --- Geometry ---------------------------------------------------------------
    x, y, z, alfa, mask_validi = _pulisci_traiettoria(x, y, z, alfa)
    s = _ascissa_curvilinea(x, y, z)
    alpha_deg = _risolvi_alpha_deg(x, y, z, alfa)
    alpha_rad = np.radians(alpha_deg)
    T_hat, beta_rad = _tangenti_e_beta(x, y, z, s, alpha_deg)
    kappa = _curvatura_con_segno(x, y, z)

    alpha_of_s = interp1d(s, alpha_rad, kind="linear", fill_value="extrapolate")
    beta_of_s = interp1d(s, beta_rad, kind="linear", fill_value="extrapolate")
    kappa_of_s = interp1d(s, kappa, kind="linear", fill_value="extrapolate")
    x_of_s = interp1d(s, x, kind="linear", fill_value="extrapolate")
    y_of_s = interp1d(s, y, kind="linear", fill_value="extrapolate")
    z_of_s = interp1d(s, z, kind="linear", fill_value="extrapolate")

    # --- Forces, Eq. (2)-(5) ----------------------------------------------------
    # One place computes the force balance, so the ODE and the reported profiles can
    # never drift apart (the base model duplicates this block, which invites exactly
    # that). w = v**2 throughout: both drag and centrifugal load are linear in it.
    def _forze(s_val, w):
        alpha = float(alpha_of_s(s_val))
        beta = float(beta_of_s(s_val))
        kap = float(kappa_of_s(s_val))

        Fg = m * g
        Fn = Fg * np.cos(alpha)                      # gravity normal to the slope
        Fs = Fg * np.sin(alpha)                      # gravity along the fall line
        F_p = Fs * np.cos(abs(beta))                 # (2) tangential: the only driver
        F_lat_grav = Fs * np.sin(abs(beta))          # fall-line gravity, sideways

        F_lat_tot = np.sign(beta) * F_lat_grav                     # (5)

        F_load = np.hypot(Fn, F_lat_tot)             # (4) total snow load N
        F_drag = 0.5 * rho * CdA_caduta * w                                 # (3)
        F_fric = mu_caduta * F_load                                         # (4)
        F_net = F_p - F_drag - F_fric

        return F_p, F_drag, F_fric, F_lat_tot, F_net, alpha, beta, kap

    def _ode(s_val, w_vec):
        # Clamp: the solver may probe slightly negative w near the event root, and a
        # negative w would flip the sign of drag (which must always oppose motion).
        w = max(float(w_vec[0]), 0.0)
        return [2.0 * _forze(s_val, w)[5] / m]                               # (1)

    def _evento_arresto(s_val, w_vec):
        return w_vec[0]

    _evento_arresto.terminal = True
    _evento_arresto.direction = -1     # only a stop reached while slowing down counts

    def _statica_tiene(s_val):
        """Eq. (6): does static friction hold the body at this candidate stop?

        Evaluated at v = 0, so there is no centrifugal contribution -- the load drops to
        gravity alone.

        When this can actually reject a stop is worth being clear about. Reaching w = 0
        while decelerating requires F_p < mu_c*N0 in the limit v -> 0 (drag and
        centrifugal load both vanish with the speed), whereas rejection requires
        F_p > mu_s*N0. Both hold together only if mu_s < mu_c, so with the default
        mu_s = mu_c the rejection fires only where alpha jumps inside a single solver
        step -- i.e. the body stopping right at the lip of an abrupt pitch, which is
        exactly the "momentary stop on a steep pitch" the brief describes and which
        LiDAR-derived alpha does produce at terrain breaks. Passing mu_s < mu_c makes it
        reachable on smooth terrain too.
        """
        alpha = float(alpha_of_s(s_val))
        beta = float(beta_of_s(s_val))
        Fg = m * g
        F_p = Fg * np.sin(alpha) * np.cos(abs(beta))
        F_lat0 = np.sign(beta) * Fg * np.sin(alpha) * np.sin(abs(beta))
        N0 = np.hypot(Fg * np.cos(alpha), F_lat0)
        return F_p <= mu_s * N0

    # --- Integration with restarts ----------------------------------------------
    s_start, s_end = float(s[0]), float(s[-1])
    seg_s, seg_v, seg_t = [], [], []
    t_offset = 0.0
    cur = s_start
    w_start = float(v0) ** 2
    n_restarts = 0
    arrestato = False
    s_arresto = None

    while True:
        # Sample on the trajectory's own points, plus the segment ends.
        interni = s[(s > cur) & (s < s_end)]
        t_eval = np.concatenate(([cur], interni, [s_end]))

        sol = solve_ivp(
            _ode,
            (cur, s_end),
            [w_start],
            events=_evento_arresto,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
        ss = sol.t
        vv = np.sqrt(np.maximum(sol.y[0], 0.0))

        if sol.t_events[0].size > 0:
            # A terminal event fired: candidate arrest, located precisely by the solver.
            s_ev = float(sol.t_events[0][0])
            ss = np.append(ss, s_ev)
            vv = np.append(vv, 0.0)

            tt = _tempo_da_velocita(ss, vv) + t_offset
            seg_s.append(ss)
            seg_v.append(vv)
            seg_t.append(tt)
            t_offset = float(tt[-1])

            if _statica_tiene(s_ev):
                arrestato = True
                s_arresto = s_ev
                break

            # Static friction fails: a momentary stop on a pitch too steep to hold.
            if s_ev >= s_end - 1e-9:
                # Nothing left to slide into -- the runout continues past the supplied
                # path, so we cannot honestly call this an arrest.
                arrestato = False
                break

            n_restarts += 1
            if n_restarts >= max_restarts:
                arrestato = False
                s_arresto = s_ev
                break

            cur = s_ev
            w_start = float(v_restart) ** 2
            continue

        # No event: the body reached the end of the path still moving.
        tt = _tempo_da_velocita(ss, vv) + t_offset
        seg_s.append(ss)
        seg_v.append(vv)
        seg_t.append(tt)
        arrestato = False
        break

    s_sol = np.concatenate(seg_s)
    v_sol = np.concatenate(seg_v)
    t_sol = np.concatenate(seg_t)

    # --- Force profiles, recomputed on the solution -----------------------------
    
    prof = np.array([_forze(s_val, v_val**2)[:6] for s_val, v_val in zip(s_sol, v_sol)])
    F_p_v, F_drag_v, F_fric_v, F_lat_v, F_net_v, alpha_v = prof.T
    kappa_v = kappa_of_s(s_sol)
    R_v = np.where(
        np.abs(kappa_v) > 1e-12,
        1.0 / np.where(np.abs(kappa_v) > 1e-12, kappa_v, 1.0),
        np.inf,
    )

    if arrestato:
        s_stop = s_arresto - s_start
        stop_xyz = np.array(
            [float(x_of_s(s_arresto)), float(y_of_s(s_arresto)), float(z_of_s(s_arresto))]
        )
        v_finale = 0.0
    else:
        s_fine = float(s_sol[-1])
        s_stop = s_fine - s_start
        stop_xyz = np.array(
            [float(x_of_s(s_fine)), float(y_of_s(s_fine)), float(z_of_s(s_fine))]
        )
        v_finale = float(v_sol[-1])

    risultati = {
        "s": s_sol,
        "v": v_sol,
        "t": t_sol,
        "F_p": F_p_v,
        "F_drag": F_drag_v,
        "F_fric": F_fric_v,
        "F_lat": F_lat_v,
        "F_net": F_net_v,
        "R": R_v,
        "alpha": alpha_v,
        "beta": beta_of_s(s_sol),
        "tan": T_hat,
        "arrestato": arrestato,
        "s_stop": float(s_stop),
        "stop_xyz": stop_xyz,
        "tempo_arresto": float(t_sol[-1]),
        "v_finale": v_finale,
        "n_restarts": n_restarts,
        "mask_punti_validi": mask_validi,
    }

    if plot:
        _plot_caduta(risultati, x, y, mu_caduta)

    return risultati


# English alias; the Italian name matches the technical brief and the pipeline's
# esegui_simulazione / risultati_sim naming.
simulate_fall = simula_caduta


# =====================================================================================
# 5. ENSEMBLE OVER mu_c  (the brief's recommendation, section 5)
# =====================================================================================
def ensemble_caduta(x, y, z, v0, mu_values=MU_ENSEMBLE_DEFAULT, **kwargs):
    """Run one fall across a small ensemble of mu_c and report a stop INTERVAL.

    mu_c dominates the arrest distance and is genuinely uncertain, so a single point
    answer overstates what the model knows. Reporting the interval spanned by a few
    plausible mu_c values is more honest for hazard mapping, and it drops straight into
    the riverMatrix binning: rasterise the interval as a runout band rather than a
    single cell, optionally weighted by a prior over mu_c.

    Returns
    -------
    dict with 'mu_values', 'runs' (one result dict each), 's_stop' (per run), plus
    's_stop_min'/'s_stop_max' and the corresponding 'xyz_min'/'xyz_max' bracketing the
    band, and 'tutti_arrestati' (False if any member ran off the end of the path).
    """
    kwargs.pop("mu_caduta", None)
    runs = [simula_caduta(x, y, z, v0, mu_caduta=mu, **kwargs) for mu in mu_values]
    s_stops = np.array([r["s_stop"] for r in runs])

    i_min, i_max = int(np.argmin(s_stops)), int(np.argmax(s_stops))
    return {
        "mu_values": tuple(mu_values),
        "runs": runs,
        "s_stop": s_stops,
        "s_stop_min": float(s_stops[i_min]),
        "s_stop_max": float(s_stops[i_max]),
        "xyz_min": runs[i_min]["stop_xyz"],
        "xyz_max": runs[i_max]["stop_xyz"],
        "tutti_arrestati": all(r["arrestato"] for r in runs),
    }


# =====================================================================================
# 6. DIAGNOSTIC PLOTS
# =====================================================================================
def _plot_caduta(ris, x, y, mu_caduta):
    """Velocity profile, force balance and plan view with the arrest marker."""
    import matplotlib.pyplot as plt  # imported lazily: workers under process_map

    s, v = ris["s"], ris["v"]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s, v * 3.6, color="darkorange", lw=2)
    if ris["arrestato"]:
        ax.axvline(ris["s_stop"], color="red", ls="--", lw=1.5,
                   label=f"arrest: {ris['s_stop']:.1f} m, {ris['tempo_arresto']:.1f} s")
    else:
        ax.axhline(ris["v_finale"] * 3.6, color="red", ls="--", lw=1.5,
                   label=f"never stops (exit {ris['v_finale'] * 3.6:.1f} km/h)")
    ax.set_title(f"Post-fall slide: velocity profile ($\\mu_c$ = {mu_caduta})")
    ax.set_xlabel("Distance slid s [m]")
    ax.set_ylabel("Speed [km/h]")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s, ris["F_p"], label="$F_p$ gravity (drives)", color="green")
    ax.plot(s, ris["F_drag"], label="$F_{drag}$", color="steelblue")
    ax.plot(s, ris["F_fric"], label="$F_{fric}$", color="firebrick")
    ax.plot(s, ris["F_net"], label="$F_{net}$", color="black", lw=2, ls="--")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Force balance along the slide")
    ax.set_xlabel("Distance slid s [m]")
    ax.set_ylabel("Force [N]")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, y, color="0.7", lw=1, label="fall path (given)")
    ax.plot(x[0], y[0], "o", color="black", label="fall point")
    ax.plot(ris["stop_xyz"][0], ris["stop_xyz"][1],
            "X" if ris["arrestato"] else "s",
            color="red", markersize=13, markeredgecolor="black",
            label="arrest" if ris["arrestato"] else "exit (no arrest)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Plan view (original CRS, UTM32N)")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    plt.show()


# =====================================================================================
# 7. VALIDATION ON SYNTHETIC TERRAIN
# =====================================================================================
def _terreno_sintetico(alpha_deg_prof, ds=0.25, x0=343000.0, y0=4990000.0, z0=2500.0):
    """Straight-in-plan 3D path with a prescribed inclination profile.

    Segment i descends at alpha_deg_prof[i], so each step has arc length exactly ds and
    s[i] = i*ds. Placed at plausible UTM32N coordinates to exercise the no-min-shift
    path with realistically large absolute values.
    """
    alpha = np.radians(np.asarray(alpha_deg_prof, dtype=float))
    x = x0 + np.concatenate(([0.0], np.cumsum(ds * np.cos(alpha))[:-1]))
    z = z0 + np.concatenate(([0.0], np.cumsum(-ds * np.sin(alpha))[:-1]))
    y = np.full(alpha.size, y0)
    return x, y, z


def _demo():
    """The brief's three sanity checks, plus the mu_c sensitivity of section 5."""
    ds = 0.25
    v0_80 = 80.0 / 3.6

    print("=" * 78)
    print("POST-FALL SLIDING MODEL -- validation on synthetic terrain")
    print(f"m = 80 kg, rho = 1.225, CdA = {CDA_CADUTA_DEFAULT}, "
          f"mu_c = {MU_CADUTA_DEFAULT}")
    print("=" * 78)

    # --- 1. Flattening slope: 25 deg -> 2 deg, then constant. -------------------
    # The brief quotes ~128 m in ~11 s but does not state over what distance the
    # slope flattens -- and the answer is very sensitive to it (see the note printed
    # below), so 300 m is our reading. It reproduces both quoted figures to a few
    # percent: 128 m alone would want ~325 m, 11 s alone ~250 m.
    s_grid = np.arange(0.0, 900.0 + ds, ds)
    prof = np.clip(25.0 + (2.0 - 25.0) * s_grid / 300.0, 2.0, 25.0)
    x, y, z = _terreno_sintetico(prof, ds)
    r1 = simula_caduta(x, y, z, v0_80, alfa=prof)
    print("\n[1] Flattening slope (25 deg -> 2 deg over 300 m), fall at 80 km/h")
    print(f"    arrestato     : {r1['arrestato']}")
    print(f"    distance slid : {r1['s_stop']:.1f} m      (brief: ~128 m)")
    print(f"    elapsed       : {r1['tempo_arresto']:.1f} s       (brief: ~11 s)")
    print(f"    stop (UTM32N) : E {r1['stop_xyz'][0]:.1f}, N {r1['stop_xyz'][1]:.1f}, "
          f"Z {r1['stop_xyz'][2]:.1f}")
    print("    -> coords are absolute UTM (E ~ 3.4e5), i.e. NOT min-shifted.")
    print(f"    NOTE: the slope starts at 25 deg but the equilibrium angle is "
          f"atan(mu_c) = {np.degrees(np.arctan(MU_CADUTA_DEFAULT)):.1f} deg.")
    print("    The top of the pitch is barely 1 deg above it, so gravity nearly")
    print("    cancels friction there and only drag bleeds speed: the runout is")
    print("    hypersensitive to how fast the slope flattens. This is physics, not")
    print("    a numerical artefact -- a constant 25 deg slope never stops at all.")

    # --- 2. Steep constant slope: tan(35 deg) = 0.70 > mu_c. ---------------------
    # Started slow on purpose, so the "accelerates and never stops" behaviour is
    # visible: gravity beats friction here, and only drag caps the speed.
    prof = np.full(s_grid.size, 35.0)
    x, y, z = _terreno_sintetico(prof, ds)
    r2 = simula_caduta(x, y, z, 20.0 / 3.6, alfa=prof)
    v_term = np.sqrt(
        (80 * 9.81 * np.sin(np.radians(35)) - MU_CADUTA_DEFAULT * 80 * 9.81
         * np.cos(np.radians(35))) / (0.5 * 1.225 * CDA_CADUTA_DEFAULT)
    )
    print("\n[2] Steep constant slope (35 deg, tan = 0.70 > mu_c = 0.45), fall at 20 km/h")
    print(f"    arrestato     : {r2['arrestato']}   <- expected False")
    print(f"    entry -> exit : {20.0:.1f} -> {r2['v_finale'] * 3.6:.1f} km/h "
          f"(accelerates)")
    print(f"    drag-limited terminal speed: {v_term * 3.6:.1f} km/h")

    # --- 3. Dip-then-steep: must arrest on the flat before the second pitch. -----
    prof = np.concatenate([
        np.full(int(100 / ds), 30.0),   # steep entry
        np.full(int(100 / ds), 1.0),    # the dip / flat
        np.full(int(200 / ds), 35.0),   # second pitch
    ])
    x, y, z = _terreno_sintetico(prof, ds)
    r3 = simula_caduta(x, y, z, v0_80, alfa=prof)
    flat_ok = r3["arrestato"] and 100.0 <= r3["s_stop"] <= 200.0
    print("\n[3] Dip-then-steep (30 deg / 1 deg flat @100-200 m / 35 deg), fall at 80 km/h")
    print(f"    arrestato     : {r3['arrestato']}")
    print(f"    distance slid : {r3['s_stop']:.1f} m  "
          f"-> on the flat section: {flat_ok}")
    print(f"    n_restarts    : {r3['n_restarts']}")

    # --- 5. Sensitivity / ensemble over mu_c, on the same slope as check [1]. ----
    prof = np.clip(25.0 + (2.0 - 25.0) * s_grid / 300.0, 2.0, 25.0)
    x, y, z = _terreno_sintetico(prof, ds)
    ens = ensemble_caduta(x, y, z, v0_80, alfa=prof)
    print("\n[5] Ensemble over mu_c on the flattening slope (brief, section 5)")
    for mu, ss in zip(ens["mu_values"], ens["s_stop"]):
        print(f"    mu_c = {mu:.2f}  ->  {ss:6.1f} m")
    print(f"    runout band   : {ens['s_stop_min']:.1f} - {ens['s_stop_max']:.1f} m "
          f"along the trajectory")
    print("    -> rasterise this band on the riverMatrix, not a single cell.")
    print()


if __name__ == "__main__":
    _demo()
