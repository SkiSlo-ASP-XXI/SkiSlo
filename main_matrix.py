import argparse, gc, random, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



from trajectory.trajectoryFromPorte import trajectoryLoader

from skier_model_py.physical_model import esegui_simulazione
from skier_model_py.fall_simulation import simula_caduta
from trajectory.tangent_derivative import obtain_inclination, obtain_slope_borders, sample_surface_z, unit_tangents

from tqdm.contrib.concurrent import process_map
from scipy.ndimage import convolve
from scipy.spatial import cKDTree

"""Best-fit plane of a LAS/LAZ point cloud.
 
The plane is returned in implicit form
 
    a*x + b*y + c*z + d = 0,     with (a, b, c) unit and c >= 0 (upward normal)
 
so that it plugs straight into the steepest-descent formula:
 
    d_hat ~ ( a*c, b*c, -(a**2 + b**2) )
"""
 
from typing import Optional, Sequence, Union
 
import numpy as np
import laspy
 
 
# ----------------------------------------------------------------------
# core fit (works on an in-memory (N, 3) array)
# ----------------------------------------------------------------------
def _fit_plane_points(P: np.ndarray, mode: str = "orthogonal"):
    """Return (normal, d, centroid) for the best-fit plane of P (N, 3)."""
    centroid = P.mean(axis=0)
    Q = P - centroid                      # centering is essential with UTM coords
 
    if mode == "orthogonal":
        # Total least squares: the normal is the direction of least variance,
        # i.e. the right-singular vector of the smallest singular value.
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        normal = Vt[-1]
    elif mode == "vertical":
        # Ordinary least squares on z:  z = p*x + q*y (+0, since centered)
        A = Q[:, :2]
        pq, *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
        normal = np.array([-pq[0], -pq[1], 1.0])
        normal /= np.linalg.norm(normal)
    else:
        raise ValueError("mode must be 'orthogonal' or 'vertical'")
 
    if normal[2] < 0:                     # keep the normal pointing up
        normal = -normal
 
    d = -float(normal @ centroid)
    return normal, d, centroid
 
 
def _residuals(P: np.ndarray, normal: np.ndarray, d: float):
    """Signed orthogonal distances of P from the plane."""
    return P @ normal + d
 
 
# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def find_plane(
    las_path: str,
    mode: str = "orthogonal",
    classification: Optional[Union[int, Sequence[int]]] = None,
    max_points: Optional[int] = 2_000_000,
    trim_sigma: Optional[float] = None,
    trim_iters: int = 3,
    seed: int = 0,
):
    """Fit the plane that best approximates the point cloud stored in `las_path`.
 
    Parameters
    ----------
    las_path : str
        Path to a .las / .laz file.
    mode : {'orthogonal', 'vertical'}
        'orthogonal' minimises the mean squared *perpendicular* distance
        (total least squares / PCA) -- the right choice if you want the plane
        that geometrically fits the cloud best, and it is invariant to how the
        cloud is rotated.
        'vertical' minimises the mean squared error *along z*, i.e. it fits
        z = p*x + q*y + r. This is the classic regression plane and is usually
        what you want for a terrain surface, since the cloud is a height field
        and the error you care about is an elevation error.
    classification : int or sequence of int, optional
        Keep only these LAS classification codes (e.g. 2 = ground).
    max_points : int, optional
        Random subsample above this size. The fit only needs second-order
        moments, so a few 10^5 points already give a converged answer.
    trim_sigma : float, optional
        If given, refit `trim_iters` times, each time discarding points whose
        residual exceeds `trim_sigma` robust sigmas (MAD-based). Cheap way to
        stop trees / lifts / people from tilting the plane.
    seed : int
        Seed of the subsampling RNG (reproducibility).
 
    Returns
    -------
    dict with keys
        normal          (3,) unit normal (a, b, c), c >= 0
        d               scalar, so that a*x + b*y + c*z + d = 0
        coeffs          (a, b, c, d)
        centroid        (3,) centroid of the points actually used
        z_of_xy         callable (x, y) -> z on the plane
        downhill        (3,) unit vector of steepest descent inside the plane
        slope_deg       slope angle of the plane below horizontal
        mse_orth        mean squared orthogonal distance  [m^2]
        rmse_orth       sqrt of the above                 [m]
        mse_vert        mean squared vertical (z) error   [m^2]
        rmse_vert       sqrt of the above                 [m]
        n_points        number of points used in the final fit
    """
    las = laspy.read(las_path)
    P = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
 
    if classification is not None:
        keep = np.isin(np.asarray(las.classification), np.atleast_1d(classification))
        P = P[keep]
 
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 3:
        raise ValueError(f"{las_path}: only {len(P)} usable points, need at least 3.")
 
    if max_points is not None and len(P) > max_points:
        idx = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
        P = P[idx]
 
    normal, d, centroid = _fit_plane_points(P, mode)
 
    if trim_sigma is not None:
        for _ in range(trim_iters):
            r = _residuals(P, normal, d)
            sigma = 1.4826 * np.median(np.abs(r - np.median(r)))   # robust std
            if sigma <= 0:
                break
            keep = np.abs(r) <= trim_sigma * sigma
            if keep.sum() < max(3, 0.1 * len(P)):
                break
            P = P[keep]
            normal, d, centroid = _fit_plane_points(P, mode)
 
    a, b, c = normal
    r = _residuals(P, normal, d)
    mse_orth = float(np.mean(r ** 2))
    # vertical error: the orthogonal distance divided by cos(tilt) = |c|
    mse_vert = float(np.mean((r / c) ** 2)) if abs(c) > 1e-12 else np.inf
 
    h = a * a + b * b
    if h > 1e-24:
        downhill = np.array([a * c, b * c, -h]) / np.sqrt(h * (h + c * c))
    else:
        downhill = np.zeros(3)           # horizontal plane: no descent direction
 
    return {
        "normal": normal,
        "d": float(d),
        "coeffs": (float(a), float(b), float(c), float(d)),
        "centroid": centroid,
        "z_of_xy": lambda x, y: -(a * np.asarray(x) + b * np.asarray(y) + d) / c,
        "downhill": downhill,
        "slope_deg": float(np.degrees(np.arccos(np.clip(abs(c), -1, 1)))),
        "mse_orth": mse_orth,
        "rmse_orth": float(np.sqrt(mse_orth)),
        "mse_vert": mse_vert,
        "rmse_vert": float(np.sqrt(mse_vert)),
        "n_points": int(len(P)),
    }

def downhill(a, b, c):
    if c < 0: a, b, c = -a, -b, -c
    v = np.array([a, b, -(a*a + b*b)/c])   # unnormalized, no square roots
    return v / np.linalg.norm(v)

def get_bump_coeff(alfa, tan, der_x, der_y, sw=None, tol:float=1e-3):

    if sw is None:
        sw:int = int(0.01*len(alfa))  
    sw:int = ((sw - sw % 2) // 2) if sw > 10 else 15
        
    #calculate vector normal to the tangent vector
    
    normal = np.concatenate([tan[:, 1::-1] * np.array([-1, 1]), tan[:, 2:3]], axis=1)
    
    gamma = np.degrees(np.atan((der_x * normal[:, 0] + der_y * normal[:, 1])/(np.linalg.norm(normal, axis=1) + 1e-9)))

    der_gamma = np.gradient(gamma)
    
    der_sw_gamma = np.zeros_like(der_gamma)

    N = len(der_gamma)

    # cumsum con 0 davanti: c[k] = somma di der_gamma[:k]
    c = np.concatenate(([0.0], np.cumsum(der_gamma)))

    # per ogni j, la finestra è [max(0, j-sw), min(N, j+sw)]
    j = np.arange(N)
    
    der_sw_gamma = (c[np.minimum(j + sw, N)] - c[np.maximum(j - sw, 0)]) / (np.minimum(j + sw, N) - np.maximum(j - sw, 0))

    N = len(der_sw_gamma)

    # 1) segno "robusto": tratta i quasi-zeri come 0 così non generi falsi cambi (0 → +) o (+ → 0) su un unico crossing reale
    sign = np.sign(der_sw_gamma)
    sign[np.abs(der_sw_gamma) < tol] = 0

    # 2) cambio di segno stretto tra i e i+1: solo + <-> - (0 non conta)
    # 3) cumsum con uno 0 davanti → c[k] = numero di crossing prima dell'indice k
    c = np.concatenate(([0], np.cumsum(((sign[:-1] * sign[1:]) < 0 ).astype(np.int64))))   # shape (N,)

    # 4) conteggio in finestra [j-sw, j+sw], con clipping automatico sui bordi
    j = np.arange(N)
    count_dgamma_zeros = (c[np.minimum(j + sw, N - 1)] - c[np.maximum(j - sw, 0)]).astype(float)


    #Now do the same on alfa
    alpha = np.degrees(np.atan((der_x * tan[:, 0] + der_y * tan[:, 1])
                                       / (np.linalg.norm(tan, axis=1) + 1e-9)))
    der_alfa = np.gradient(alpha)

    der_sw_alfa = np.zeros_like(der_alfa)

    for j in range(sw, len(der_alfa)-sw):
        der_sw_alfa[j] = np.mean(der_alfa[j-sw:j+sw])
        
    for j in range(sw):
        der_sw_alfa[j] = np.mean(der_alfa[0:j+sw])
            
    for j in range(len(der_alfa)-sw, len(der_alfa)):
        der_sw_alfa[j] = np.mean(der_alfa[j-sw:len(der_alfa)])
    
    N = len(der_sw_alfa)

    mask = (np.abs(der_sw_alfa) < tol).astype(np.int64)

    # "start interno alla finestra" = posizioni i>a con mask[i]=1 & mask[i-1]=0
    # → si contano sommando sulla shifted-diff nell'intervallo (a, b)
    # "start di bordo" = mask[a]=1 (perché il tuo prepend=0 lo tratta come start)

    # transizioni 0->1 nell'array intero, riferite alla posizione di destinazione
    inner_starts = np.zeros(N, dtype=np.int64)
    inner_starts[1:] = ((mask[1:] == 1) & (mask[:-1] == 0)).astype(np.int64)

    c = np.concatenate(([0], np.cumsum(inner_starts)))

    j = np.arange(N)
    # start interni nella finestra (a, b): c[b] - c[a+1]... ma serve gestire a==b
    # più semplice: start interni in [a+1, b) = c[b] - c[a+1]
    # più il bordo: mask[a] == 1 (conta come "0->1" per via del prepend=0)
    count_dalfa_zeros = (c[np.minimum(j + sw, N - 1)] - c[np.maximum(j - sw, 0)]).astype(float)
    
    
    haz_coeff = count_dalfa_zeros+count_dgamma_zeros

    haz_coeff = (haz_coeff-np.min(haz_coeff))/(np.max(haz_coeff)-np.min(haz_coeff) + 1e-9)

    return haz_coeff

def set_seed(seed:int) -> int:
    np.random.seed(seed)
    random.seed(seed)
    return seed

def get_descent_direction(path_to_las:str):
    plane_res = find_plane(path_to_las)
    a,b,c = plane_res['normal']
    desc_vect = downhill(a,b,c)
    return desc_vect

def fill_nan_nearest(a:np.ndarray) -> np.ndarray:
    """Sostituisce i NaN/None con il valore valido più vicino (a parità di distanza vince quello a sinistra)."""
    a = np.asarray(a, dtype=float)  # None -> NaN
    valid = np.flatnonzero(~np.isnan(a))
    if valid.size == 0:
        return np.zeros_like(a)
    if valid.size == a.size:
        return a

    idx = np.arange(a.size)
    pos = np.searchsorted(valid, idx)
    left = valid[np.clip(pos - 1, 0, valid.size - 1)]
    right = valid[np.clip(pos, 0, valid.size - 1)]
    nearest = np.where(np.abs(idx - left) <= np.abs(idx - right), left, right)
    return a[nearest]

class SnowDepthTree:
    """Ricerca del punto più vicino nel piano (x, y) per la profondità della neve.

    La nuvola della differenza di neve ha ~10^7 punti, quindi si tiene solo lo
    stretto necessario: le due coordinate orizzontali in float64 (le UTM hanno
    bisogno della precisione) e la profondità in float32. La z viene ignorata.
    """

    # La colonna della profondità cambia nome a seconda dell'export
    _DEPTH_ALIASES = ("relative height", "relative_height", "depth")

    def __init__(self, xy:np.ndarray, depth:np.ndarray):
        self.xy = np.ascontiguousarray(xy, dtype=np.float64)
        self.depth = np.ascontiguousarray(depth, dtype=np.float32)
        if self.xy.shape != (self.depth.size, 2):
            raise ValueError(f"xy {self.xy.shape} non compatibile con depth {self.depth.shape}")
        self.tree = cKDTree(self.xy)

    @classmethod
    def from_dataframe(cls, df) -> "SnowDepthTree":
        """Costruisce l'albero da un dataframe con colonne x, y, z, "Relative height".

        La z viene scartata e "Relative height" diventa "depth". Le colonne sono
        copiate fuori dal dataframe, che può quindi essere liberato subito dopo.
        """
        # CloudCompare scrive l'header come "//X,Y,Z,Relative height"
        cols = {str(c).strip().lstrip("/").lower(): c for c in df.columns}
        x_col, y_col = cols.get("x"), cols.get("y")
        depth_col = next((cols[alias] for alias in cls._DEPTH_ALIASES if alias in cols), None)
        if x_col is None or y_col is None or depth_col is None:
            raise ValueError(f"attese le colonne x, y e depth, trovate {list(df.columns)}")

        xy = np.column_stack((df[x_col].to_numpy(dtype=np.float64),
                              df[y_col].to_numpy(dtype=np.float64)))
        depth = df[depth_col].to_numpy(dtype=np.float32)
        return cls(xy, depth)

    def query(self, x, y, max_distance:Optional[float]=None, fill_value:float=np.nan, workers:int=-1):
        """Profondità del campione più vicino a ogni (x, y).

        Accetta scalari o array (restituisce rispettivamente un float o un array).
        I punti più lontani di `max_distance` da qualsiasi campione ricevono
        `fill_value`; senza `max_distance` si prende sempre il più vicino.
        """
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                     np.asarray(y, dtype=np.float64))
        pts = np.column_stack((xa.ravel(), ya.ravel()))
        upper = np.inf if max_distance is None else float(max_distance)
        _, idx = self.tree.query(pts, distance_upper_bound=upper, workers=workers)

        out = np.full(idx.shape, fill_value, dtype=np.float64)
        found = idx < self.depth.size   # fuori dal raggio cKDTree restituisce n
        out[found] = self.depth[idx[found]]

        if xa.ndim == 0:
            return float(out[0])
        return out.reshape(xa.shape)

def return_inclination(x_utm:np.ndarray, y_utm:np.ndarray, z_traj:np.ndarray, path_to_las:str, desc_vect: np.ndarray):
    alpha_deg, grads, real_z = obtain_inclination(x_utm,y_utm,path_to_las, desc_vect)
    alpha_deg = -alpha_deg
    N = len(x_utm)
    alpha_deg = fill_nan_nearest(alpha_deg)
    alpha_deg[N-1] = alpha_deg[N-2]
    alpha_deg[0] = alpha_deg[1] # Riempimento bordo iniziale

    return alpha_deg, grads, real_z
        
# ======================================================================
# TEMPORANEO: confronto fra i tre modi di calcolare alpha.
# Da rimuovere una volta scelto il metodo definitivo.
# ======================================================================
def chord_direction(df) -> np.ndarray:
    """Versore 3D dalla partenza all'arrivo della traiettoria.

    È il `vec_ref` che physical_model.esegui_simulazione calcolava nel blocco ora
    commentato (righe 69-72): una singola retta dalla prima all'ultima porta, che
    non sa nulla del terreno locale ma solo della discesa complessiva. È immune
    alla traslazione che esegui_simulazione applica alle coordinate (è una
    differenza), quindi si può calcolare qui sul df originale.

    La componente verticale conta: esegui_simulazione ne ricava gamma, quindi la z
    dev'essere la stessa passata alla simulazione ("Quota Orto. [m]").
    """
    v = np.array([df["Est [m]"].values[-1]        - df["Est [m]"].values[0],
                  df["Nord [m]"].values[-1]       - df["Nord [m]"].values[0],
                  df["Quota Orto. [m]"].values[-1] - df["Quota Orto. [m]"].values[0]])
    return v / (np.linalg.norm(v) + 1e-12)


def alpha_from_direction(u, grads:np.ndarray) -> np.ndarray:
    """alpha [deg] = -atan(grad(z) . u_hat), con u_hat orizzontale e unitario.

    Non serve rifare il fit dei piani sul .las: `grads` (dz/dx, dz/dy per punto) è
    già quello calcolato da tangent_derivatives, cambia solo la direzione su cui lo
    si proietta. `u` può essere una singola direzione (2,)/(3,) oppure una per
    punto (N, 2)/(N, 3); si usa solo la parte orizzontale, normalizzata qui perché
    grad(z) . u_hat è la salita per metro percorso (= tan(alpha)) solo se u_hat è
    orizzontale e unitario. La post-elaborazione replica return_inclination.
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))[:, :2]
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    der = grads[:, 0] * u[:, 0] + grads[:, 1] * u[:, 1]

    alpha_deg = -np.degrees(np.arctan(der))
    alpha_deg = fill_nan_nearest(alpha_deg)
    alpha_deg[-1] = alpha_deg[-2]
    alpha_deg[0] = alpha_deg[1]
    return alpha_deg

# Metodo di calcolo di alpha effettivamente usato: è l'UNICO che entra nella
# simulazione e nel calcolo degli hazard coefficient. Gli altri due vengono
# calcolati e disegnati solo come riferimento in alpha_comparison.png.
ALPHA_METHOD = 'corda'

# Etichette dei tre metodi, nell'ordine in cui vengono disegnati.
ALPHA_LABELS = {
    'max_pendenza': 'Max pendenza (piano globale) - solo riferimento',
    'tangente':     'Tangente traiettoria (locale) - solo riferimento',
    'corda':        'Corda inizio-fine - USATO in simulazione e hazard',
}
ALPHA_COLORS = {'max_pendenza': 'tab:blue', 'tangente': 'tab:orange', 'corda': 'tab:green'}

def plot_alpha_comparison(alphas_all:list, save_path:str="alpha_comparison.png"):
    """Confronta i tre alpha: max pendenza, tangente locale, corda inizio-fine.

    `alphas_all` è la lista (una voce per traiettoria) dei dizionari prodotti da
    `_simulate`, con una chiave per metodo.
    """
    keys = list(ALPHA_LABELS)
    ref = ALPHA_METHOD   # gli scarti si misurano rispetto all'alpha davvero usato

    fig, (ax_prof, ax_diff, ax_sc) = plt.subplots(1, 3, figsize=(18, 5))

    for k in keys:
        ax_prof.plot(alphas_all[0][k], color=ALPHA_COLORS[k], linewidth=1.0, label=ALPHA_LABELS[k])
    ax_prof.set_xlabel('Indice punto'); ax_prof.set_ylabel('alpha [deg]')
    ax_prof.set_title('Profilo di inclinazione (traiettoria 0)')
    ax_prof.grid(alpha=0.3); ax_prof.legend(loc='best', fontsize=8)

    for k in keys:
        if k == ref:
            continue
        ax_diff.plot(alphas_all[0][k] - alphas_all[0][ref], color=ALPHA_COLORS[k], linewidth=0.8,
                     label=f'{ALPHA_LABELS[k]} - corda')
    ax_diff.axhline(0, color='black', linewidth=0.6)
    ax_diff.set_xlabel('Indice punto'); ax_diff.set_ylabel('differenza [deg]')
    ax_diff.set_title('Scarto rispetto alla corda (traiettoria 0)')
    ax_diff.grid(alpha=0.3); ax_diff.legend(loc='best', fontsize=8)

    cat = {k: np.concatenate([a[k] for a in alphas_all]) for k in keys}
    for k in keys:
        if k == ref:
            continue
        ax_sc.scatter(cat[ref], cat[k], s=2, alpha=0.2, color=ALPHA_COLORS[k], label=ALPHA_LABELS[k])
    lims = [min(v.min() for v in cat.values()), max(v.max() for v in cat.values())]
    ax_sc.plot(lims, lims, color='black', linewidth=1.0, linestyle='--', label='1:1')
    ax_sc.set_xlabel('Corda inizio-fine [deg]'); ax_sc.set_ylabel('Altro metodo [deg]')
    ax_sc.set_title(f'Tutte le traiettorie ({len(cat[ref])} punti)')
    ax_sc.grid(alpha=0.3); ax_sc.legend(loc='best', fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', format='png')

    for k in keys:
        v = cat[k]
        print(f"alpha {ALPHA_LABELS[k]:<45}: media={v.mean():6.2f}  min={v.min():6.2f}  max={v.max():6.2f} deg")
    for k in keys:
        if k == ref:
            continue
        d = cat[k] - cat[ref]
        print(f"  scarto ({k} - {ref}): media={d.mean():6.2f}  max|.|={np.abs(d).max():6.2f} deg")

def plot_stop_points(listDf:list, risSim:list, save_path:str="stop_points.png"):
    """Evidenzia dove lo sciatore si ferma lungo ogni traiettoria.

    Pannello sinistro: vista in pianta, tratto percorso in movimento vs tratto
    da fermo. Pannello destro: profilo di velocità in funzione dell'ascissa
    curvilinea, con il punto di arresto marcato.
    """
    fig, (ax_map, ax_v) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (df, res) in enumerate(zip(listDf, risSim)):
        x, y = df["Est [m]"].values, df["Nord [m]"].values
        moving, s, v = res['moving'], res['s'], res['v']
        k = int(moving.sum())  # primo indice da fermo

        lbl = {'label': 'In movimento'} if i == 0 else {}
        lbl_stop = {'label': 'Fermo'} if i == 0 else {}
        lbl_pt = {'label': 'Punto di arresto'} if i == 0 else {}

        ax_map.plot(x[:k], y[:k], color='tab:blue', linewidth=1.2, **lbl)
        ax_map.plot(x[k-1:], y[k-1:], color='tab:red', linewidth=1.2, **lbl_stop)
        ax_v.plot(s[:k], v[:k] * 3.6, color='tab:blue', linewidth=1.0, **lbl)
        ax_v.plot(s[k-1:], v[k-1:] * 3.6, color='tab:red', linewidth=1.0, **lbl_stop)

        if k < len(x):
            ax_map.plot(x[k-1], y[k-1], marker='o', color='black', markersize=6, zorder=5, **lbl_pt)
            ax_v.plot(s[k-1], v[k-1] * 3.6, marker='o', color='black', markersize=6, zorder=5, **lbl_pt)
            ax_v.axvline(s[k-1], color='tab:red', linewidth=0.6, alpha=0.4)
            print(f"Traj {i}: arresto a s = {s[k-1]:.1f} m su {s[-1]:.1f} m ({100*k/len(x):.0f}% del tracciato)")
        else:
            print(f"Traj {i}: lo sciatore arriva a fondo pista ({s[-1]:.1f} m)")

    ax_map.set_xlabel('Est [m]'); ax_map.set_ylabel('Nord [m]')
    ax_map.set_title('Punto di arresto lungo la traiettoria')
    ax_map.set_aspect('equal', adjustable='datalim')
    ax_map.grid(alpha=0.3); ax_map.legend(loc='best')

    ax_v.set_xlabel('Ascissa curvilinea s [m]'); ax_v.set_ylabel('Velocità [km/h]')
    ax_v.set_title('Profilo di velocità')
    ax_v.grid(alpha=0.3); ax_v.legend(loc='best')

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', format='png')

def _simulate(df, path_to_las:str, desc_vect):
    x, y, z = df["Est [m]"].values, df["Nord [m]"].values, df["Quota Orto. [m]"].values

    # 'corda': proiezione sulla retta partenza-arrivo. È l'alpha usato davvero, ed è
    # la stessa direzione passata a esegui_simulazione come vec_ref: alpha e vec_ref
    # vanno tenuti coerenti perché dentro esegui_simulazione compaiono accoppiati in
    # factor = cos(gamma - alpha), con gamma ricavato proprio da vec_ref.
    chord = chord_direction(df)

    # 'max_pendenza' (direzione di massima discesa del piano fittato su tutta la nuvola)
    # e 'tangente' (direzione di marcia locale) restano solo come riferimento nel plot.
    # Un solo giro sul .las: da return_inclination escono anche grads e real_z, e le
    # altre due proiezioni riusano quei gradienti senza rifittare i piani.
    alpha_pendenza, grads, real_z = return_inclination(x, y, z, path_to_las, desc_vect)
    alphas_all = {
        'max_pendenza': alpha_pendenza,
        'tangente':     alpha_from_direction(unit_tangents(np.stack([x, y], axis=1)), grads),
        'corda':        alpha_from_direction(chord, grads),
    }

    ris = esegui_simulazione(x, y, z, chord, alfa=alphas_all[ALPHA_METHOD])
    return ris, alphas_all, grads, real_z

def main(num_points:int=3_000): #To call main paste: python main_matrix.py --gates data/pointsLocationFirstCourse.csv --numTrajectories 200
    parser = argparse.ArgumentParser(description="A script that accepts keyword-like arguments.")
    #paths for loading and saving data (REQUIRED)
    parser.add_argument("--gates", type=str, help="gates_path", required=True)

    #paths for loading and saving data (OPTIONAL)
    parser.add_argument("--output", type=str, default="./outputs", help="Path to save the simulated trajectories (CSV files). If not provided, the trajectories will be saved in ./generated_trajectories.")

    #Hyperparameters for generating new trajectories (OPTIONAL)
    parser.add_argument("--numTrajectories", type=int, default=1, help="Number of new trajectories to generate.")
    parser.add_argument("--maxDistanceMeters", type=float, default=5, help="Maximum distance in meters for the generated noise.")
    parser.add_argument("--startLeft", action='store_true', help="Flag to indicate whether to start adding noise to the left of the original trajectory.")

    #parameters for plotting (OPTIONAL)
    parser.add_argument("--plot", action='store_true', help="Flag to indicate whether to plot the original and simulated trajectories.")
    parser.add_argument("--plotPath", type=str, default="./plots", help="Path to save the trajectory plots. If not provided, the plots will be saved in ./plots.")
    
    parser.add_argument("--path_to_las", type=str, default="/Users/andre/Documents/github.nosync/SkiSlo/data/surfaces/Sestriere_fotogrammetria_95000.las",
                        help="Path to the .las file containing the elevation data. If not provided, it will default to ./data/surfaces/Sestriere_fotogrammetria_95000.las.")
    parser.add_argument("--depth_paath", type=str, default="/Users/andre/Documents/github.nosync/SkiSlo/data/snow_depth/Nuvola_differenza_neve.csv",)
    
    #########################

    args = parser.parse_args()

    print(f"Gates path: {args.gates}")

    # Load gates and generate new trajectories
    loader = trajectoryLoader(args.gates)
    os.makedirs(os.path.join(args.output, "simulated_trajectories"), exist_ok=True)
    newTraj = loader.generateNewTrajectories(numTrajectories=args.numTrajectories, maxDistanceMeters=args.maxDistanceMeters, startLeft=args.startLeft)
    # loader.saveNewTrajectories(newTraj, os.path.join(args.output, "simulated_trajectories.csv"))

    # If plotting is enabled, plot the original and simulated trajectories
    if args.plot:
        os.makedirs(args.plotPath, exist_ok=True)
        loader.plotSimulatedTrajectories(newTraj, args.plotPath)

    #obtain the full (interpolated) trajectories
    listDf, risSim = [], []
    maxX, maxY, minX, minY = -np.inf, -np.inf, np.inf, np.inf

    listDf = [loader.prepareTrajectories(numPoints=num_points, gates=traj) for traj in newTraj]
    len_trajectories = []
    for i in range(len(listDf)):
        len_traj = 0
        for j in range(1, len(listDf[i])):
            dx = listDf[i]["Est [m]"].values[j] - listDf[i]["Est [m]"].values[j-1]
            dy = listDf[i]["Nord [m]"].values[j] - listDf[i]["Nord [m]"].values[j-1]
            len_traj += np.sqrt(dx**2 + dy**2)
        len_trajectories.append(len_traj)

    desc_vect = get_descent_direction(path_to_las=args.path_to_las)

    maxX, maxY = max(df["Est [m]"].max()  for df in listDf), max(df["Nord [m]"].max() for df in listDf)
    minX, minY = min(df["Est [m]"].min()  for df in listDf), min(df["Nord [m]"].min() for df in listDf)
    
    risSim = process_map(
        _simulate,
        listDf,
        [args.path_to_las] * len(listDf),
        [desc_vect] * len(listDf),
        max_workers=os.cpu_count(),
        chunksize=1,
        desc="Simulating",
        unit="traj",
    )
    risSim, alphas_all, grads, real_z  = [res[0] for res in risSim], [res[1] for res in risSim], [res[2] for res in risSim], [res[3] for res in risSim]  # Extract the simulation results and the alphas
    # Da qui in poi si usa un solo alpha, lo stesso che è entrato nella simulazione
    # (ALPHA_METHOD). Gli altri metodi restano in alphas_all solo per il confronto.
    alphas = [a[ALPHA_METHOD] for a in alphas_all]

    plot_alpha_comparison(alphas_all)   # TEMPORANEO
    plot_stop_points(listDf, risSim)

    haz_bump_coeffs = []
    
    for i in range(len(alphas)):
        haz_bump_coeffs.append(get_bump_coeff(alphas[i], risSim[i]['tan'],grads[i][:,0], grads[i][:,1], sw=30))
    

    # Post processing of the results
    haz_coeff_sim = [(np.abs(res['F_lat'])-np.min(np.abs(res['F_lat']))) / (np.max(np.abs(res['F_lat'])) - np.min(np.abs(res['F_lat']))) for res in risSim]

    # haz_coeff_incl = [(alpha-np.min(alpha)) / (np.max(alpha)-np.min(alpha)) for alpha in alphas]

    alphas_der = []
    alphas_der_sw = []

    for i in range(len(alphas)):
        alpha_der = np.gradient(alphas[i])#np.abs(np.gradient(alphas[i]))
        alpha_der_sw = np.zeros_like(alpha_der)
        
        for j in range(15, len(alpha_der)-15):
            alpha_der_sw[j] = np.mean(alpha_der[j-15:j+15])
        
        for j in range(15):
            alpha_der_sw[j] = np.mean(alpha_der[0:j+15])
            
        for j in range(len(alpha_der)-15, len(alpha_der)):
            alpha_der_sw[j] = np.mean(alpha_der[j-15:len(alpha_der)])
        
        alphas_der_sw.append(alpha_der_sw)
        alphas_der.append(alpha_der)

    #haz_coeff_incl = [(alpha-np.min(alpha)) / (np.max(alpha)-np.min(alpha)) for alpha in alphas_der_sw]

    haz_coeff_incl = [(alpha-np.min(alpha)) / (np.max(alpha)-np.min(alpha)) for alpha in alphas]


    #load the snow depth from input file
    depth_tree = None
    if args.depth_paath is not None:
        depthDf = pd.read_csv(args.depth_paath, sep=',', header=0)
        depth_tree = SnowDepthTree.from_dataframe(depthDf)
        # La nuvola pesa centinaia di MB: l'albero ha già copiato x, y e depth
        del depthDf
        gc.collect()

    haz_coeff_depth = np.array(haz_coeff_incl, dtype=float)
    if depth_tree is not None:
        for i in range(len(listDf)):
            traj = listDf[i]
            xs, ys = traj["Est [m]"].values, traj["Nord [m]"].values
            depth = depth_tree.query(xs, ys)
            depth = np.clip(depth, 0, 1)
            haz_coeff_depth[i] = 1 - depth

    # for alpha in alphas_der:
    weight = 1/4
    haz_coeff = weight* np.array(haz_bump_coeffs) + weight * np.array(haz_coeff_sim) + weight * np.array(haz_coeff_incl) + weight * np.array(haz_coeff_depth)

    # Create a river matrix with the cells of 1 meter x 1 meter and fill it with the hazard coefficients (for visualization purposes)
        
    riverMatrix = np.zeros((int(maxX - minX)+1, int(maxY - minY)+1))
    reliabilityMatrix = np.zeros((int(maxX - minX)+1, int(maxY - minY)+1))
    # The matrix is such that the zero of the x-axis is the minimum x value of the trajectories and the zero of the y-axis is the minimum y value of the trajectories. 
    # The values of the matrix are filled with the hazard coefficients, where the position in the matrix corresponds to the position in the trajectory (x, y).
    
    H, W = riverMatrix.shape
    xs, ys, hzs = [], [], []
    for df, haz in zip(listDf, haz_coeff):
        xs.append(((df["Est [m]"].values  - minX) / (maxX - minX + 1e-9) * (W - 1)).astype(np.intp))
        ys.append(((df["Nord [m]"].values - minY) / (maxY - minY + 1e-9) * (H - 1)).astype(np.intp))
        hzs.append(haz)

    flat = np.concatenate(ys) * W + np.concatenate(xs)
    riverMatrix = np.bincount(flat, weights=np.concatenate(hzs), minlength=H * W).reshape(H, W)
    reliabilityMatrix = np.bincount(flat, minlength=H * W).reshape(H, W).astype(float)

    riverMatrix = np.divide(riverMatrix, reliabilityMatrix, out=np.zeros_like(riverMatrix), where=reliabilityMatrix > 0)

    # For the few cells that are not covered by any trajectory, the risk is given by the average of the neighboring cells in any direction and diagonals.
    # However, this is done only for the cells that have at least five neighboring cells with reliability greater than zero, otherwise the cell is left as zero. 
    # This is to avoid filling in too many cells with unreliable values.
        
    # Lets use vector operations
    kernel = np.ones((3, 3), dtype=riverMatrix.dtype)

    # Sum of reliability in each 3x3 window
    rel_sum = convolve(reliabilityMatrix, kernel, mode='constant', cval=0)

    # Cells that match the original three conditions
    mask = (reliabilityMatrix == 0) & (convolve(reliabilityMatrix.astype(np.int32), kernel.astype(np.int32), mode='constant', cval=0) >= 4)

    riverMatrix = np.where(mask, convolve(riverMatrix * reliabilityMatrix, kernel, mode='constant', cval=0) /  np.where(rel_sum > 0, rel_sum, 1), riverMatrix)
    reliabilityMatrix = np.where(mask, rel_sum, reliabilityMatrix)
    
    # Building the reverse mapping
    reverseMapping = dict()
    for i, (x, y) in enumerate(zip(xs, ys)):
        for xi, yi in zip(x,y):
            if (xi, yi) not in reverseMapping:
                reverseMapping[(xi, yi)] = set()
            reverseMapping[(xi, yi)].add(i)
        
        
#======================================================================================================
    #CALCOLO TANGENTI VIE DI FUGA
#======================================================================================================

    tangents = []
    for i in range(len(risSim)):
        len_traj = len_trajectories[i]
        min_p = -5*(num_points/len_traj)
        max_p = 2*(num_points/len_traj)
        lim_var = np.percentile(haz_coeff[i], 99)
        idxs = np.argwhere(haz_coeff[i] >= lim_var)
        for max_idx in idxs:            
            #max_idx = np.argmax(haz_coeff)
            # max_idx = np.sum(max_idx)[0]
            for p in (int(min_p), int(max_p)+1):
                idx = max(0, min(len(haz_coeff[i])-1, max_idx + p))
                tangent = np.arctan2(listDf[i]["Nord [m]"].values[idx] - listDf[i]["Nord [m]"].values[idx-1], listDf[i]["Est [m]"].values[idx] - listDf[i]["Est [m]"].values[idx-1])
                tangents.append((listDf[i]["Nord [m]"].values[idx], listDf[i]["Est [m]"].values[idx], tangent))
    #FINE VIE DI FUGA    

    print("Plotting the river matrix...")
    print(f"Max X: {maxX}, Min X: {minX}, Max Y: {maxY}, Min Y: {minY}")
    print(f"Difference x: {maxX - minX}, Difference y: {maxY - minY}")
    print("Non-zero values in river matrix:", len(riverMatrix[riverMatrix > 0]))
    print("Matrix shape :", riverMatrix.shape)


    cmap = plt.get_cmap('RdYlGn_r').copy()
    cmap.set_bad(color='white')

    plt.figure(figsize=(8, 8))
    plt.imshow(np.ma.masked_equal(riverMatrix.T, 0), cmap=cmap, origin='lower')
    plt.colorbar(label='Hazard coefficient')

    # Overlay the first 10 escape tangents on the river matrix.
    # tangents[i] = (Nord, Est, angle); angle = arctan2(dNord, dEst) in real-world coords.
    #TODO: DECOMMENT
    sx = (W - 1) / (maxX - minX + 1e-9)   # Est [m]  -> column (Est) index scale
    sy = (H - 1) / (maxY - minY + 1e-9)   # Nord [m] -> row (Nord) index scale
    t = np.array([-20, 20])   # half-length of each drawn tangent, in meters
    for nord, est, angle in tangents:
        est_idx  = (est  - minX) * sx
        nord_idx = (nord - minY) * sy
        # displayed image is riverMatrix.T: plot-x = Nord index, plot-y = Est index
        plt.plot(nord_idx + np.sin(angle) * sy * t, est_idx  + np.cos(angle) * sx * t, color='blue', linewidth=1.5)
        plt.plot(nord_idx, est_idx, marker='o', color='blue', markersize=4)


    # Overlay the slope borders recovered from the --path_to_las point cloud.
    # The .las is already cropped to the slope, so the cloud's outer outline is the border.
    border_est, border_nord = obtain_slope_borders(args.path_to_las)
    border_x = (border_nord - minY) * sy   # plot-x = Nord index
    border_y = (border_est  - minX) * sx   # plot-y = Est index
    plt.scatter(border_x, border_y, color='black', s=2, marker='.',
                zorder=4, label='Slope border')

    # Overlay the gates (purple) recovered from the --gates file.
    # Drop the synthetic fakeInit/fakeFinal rows so only the real gates are shown.
    real_gates = loader.gates.drop(index=['fakeInit', 'fakeFinal'], errors='ignore')
    plt.scatter((real_gates["Nord [m]"].values - minY) * sy, (real_gates["Est [m]"].values  - minX) * sx, color='purple', s=50, marker='s',
                edgecolors='black', linewidths=0.5, zorder=5, label='Gates')

    # Mark the start and end of the trajectory (first/last interpolated points).
    start, end = listDf[0].iloc[0], listDf[0].iloc[-1]
    plt.scatter((start["Nord [m]"] - minY) * sy, (start["Est [m]"]  - minX) * sx, color='lime', s=120, marker='*',
                edgecolors='black', linewidths=0.5, zorder=6, label='Start')
    plt.scatter((end["Nord [m]"] - minY) * sy, (end["Est [m]"]  - minX) * sx, color='cyan', s=120, marker='X',
                edgecolors='black', linewidths=0.5, zorder=6, label='End')
    plt.legend(loc='best')

    plt.savefig("river_matrix.png", bbox_inches='tight', format='png')

    plt.figure(figsize=(8, 8))
    plt.imshow(np.ma.masked_equal(reliabilityMatrix.T, 0), cmap=cmap, origin='lower')
    plt.colorbar(label='Reliability')
    plt.savefig("reliability_matrix.png", bbox_inches='tight', format='png')
    
    #show results
    plt.show()


    max_HC = np.argmax(haz_coeff) #[idx_traiettoria, indx_punto_in_traiettoria]
    max_HC = (max_HC // haz_coeff.shape[1], max_HC % haz_coeff.shape[1])  # Convert flat index to 2D index
    max_HC_x = xs[max_HC[0]][max_HC[1]]
    max_HC_y = ys[max_HC[0]][max_HC[1]]

    trajectories_selected = reverseMapping.get((max_HC_x, max_HC_y), set()) 
    
    max_HC_x_continous = int(max_HC_x * (maxX - minX + 1e-9) / (W - 1) + minX)
    max_HC_y_continous = int(max_HC_y * (maxY - minY + 1e-9) / (H - 1) + minY)
    
    t = np.arange(100)   # samples along each tangent ray, in metres from the trajectory
    max_HC_tangents = []   # (traj index, point index, tangent angle) at the max-hazard cell
    rays = []              # (x_tan, y_tan) ray samples, one entry per selected trajectory
    fall_simulations = []

    for i in sorted(trajectories_selected):
        traj = listDf[i]
        idx = int(np.argmin((traj["Est [m]"].values -max_HC_x_continous)**2+(traj["Nord [m]"].values -max_HC_y_continous)**2))
        idx = max(1, min(len(traj)-2, idx))   # keep the central difference below in range
        tangent = np.arctan2(traj["Nord [m]"].values[idx+1] - traj["Nord [m]"].values[idx-1], traj["Est [m]"].values[idx+1] - traj["Est [m]"].values[idx-1])
        x_tan = traj['Est [m]'].values[idx] + np.cos(tangent) * t
        y_tan = traj['Nord [m]'].values[idx] + np.sin(tangent) * t
        z_tan = sample_surface_z(x_tan,
                                 y_tan,
                                 args.path_to_las)
        v0 = risSim[i]['v'][idx]
        rays.append((x_tan, y_tan, z_tan, v0))
        max_HC_tangents.append((i, idx, tangent))
        ret_val = simula_caduta(x_tan, y_tan, z_tan, v0)
        fall_simulations.append(ret_val)

    # Where a fallen skier ends up: each escape ray coloured by the speed along the slide,
    # over the trajectories they branch off from and the slope outline from the .las.
    fig, ax = plt.subplots(figsize=(9, 9))

    # The .las is cropped to the piste, so the outline of the cloud is the slope border.
    border_est, border_nord = obtain_slope_borders(args.path_to_las)
    ax.scatter(border_est, border_nord, color='0.8', s=2, marker='.', zorder=1,
               label='Slope border (.las)')

    for n, i in enumerate(sorted(trajectories_selected)):
        traj = listDf[i]
        ax.plot(traj["Est [m]"].values, traj["Nord [m]"].values, color='0.45', lw=0.9, zorder=2,
                label='Selected trajectories' if n == 0 else None)

    v_max_kmh = max(ris['v'].max() for ris in fall_simulations) * 3.6

    for ris, (x_tan, y_tan, z_tan, v0) in zip(fall_simulations, rays):
        # simula_caduta integrates against 3D arc length and returns only stop_xyz, so map
        # its s profile back onto the ray through the points it actually kept. The rounding
        # mirrors _pulisci_traiettoria's, so mask_punti_validi lines up with these arrays.
        keep = ris['mask_punti_validi']
        xk, yk, zk = np.round(x_tan, 2)[keep], np.round(y_tan, 2)[keep], np.round(z_tan, 2)[keep]
        s_k = np.concatenate(([0.0], np.cumsum(np.sqrt(np.diff(xk)**2 + np.diff(yk)**2 + np.diff(zk)**2))))
        x_sol, y_sol = np.interp(ris['s'], s_k, xk), np.interp(ris['s'], s_k, yk)

        # One coloured segment per step: the colour IS the speed there.
        pts = np.column_stack([x_sol, y_sol]).reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap='viridis',
                            norm=plt.Normalize(0, v_max_kmh), linewidth=3, zorder=3)
        lc.set_array(ris['v'][:-1] * 3.6)
        ax.add_collection(lc)

        ax.plot(x_sol[0], y_sol[0], marker='o', color='black', markersize=5, zorder=4)
        ax.plot(ris['stop_xyz'][0], ris['stop_xyz'][1], marker='X' if ris['arrestato'] else 's',
                color='red', markersize=11, markeredgecolor='black', zorder=5)

    fig.colorbar(lc, ax=ax, label='Speed along the slide [km/h]', shrink=0.8)

    # The markers are drawn per-ray, so label them once here instead.
    handles, _ = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls='', marker='o', color='black', label='Fall point'),
                plt.Line2D([], [], ls='', marker='X', color='red', markeredgecolor='black',
                           label='Arrest (comes to rest)'),
                plt.Line2D([], [], ls='', marker='s', color='red', markeredgecolor='black',
                           label='Still moving at slope edge')]
    ax.legend(handles=handles, loc='best', fontsize=8)

    ax.scatter(max_HC_x_continous, max_HC_y_continous, color='magenta', s=150, marker='*',
               edgecolors='black', linewidths=0.5, zorder=6)
    ax.set_xlabel("Est [m]"); ax.set_ylabel("Nord [m]")
    ax.set_title(f"Post-fall slides from the max-hazard cell (haz={haz_coeff[max_HC]:.3f})")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, ls='--', alpha=0.4)
    fig.savefig("fall_paths.png", bbox_inches='tight', format='png')
    plt.show()



if __name__ == "__main__":
    SEED:int = 23
    NUM_POINTS:int = 3_000
    assert set_seed(SEED) == SEED, "Error setting the seed"
    main()