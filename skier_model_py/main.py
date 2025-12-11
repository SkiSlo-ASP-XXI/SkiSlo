import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("error")

# ===============================
# 1. Generazione del piano
# ===============================
from geometry import genera_superficie

L = 100.0
W = 20.0
nx = 50
ny = 100
pendenza_target = 0.2

# Esempio 1: piano puro
x_grid, y_grid, Xg, Yg, Zg, h = genera_superficie(
    L=L,
    W=W,
    nx=nx,
    ny=ny,
    kind="plane",
    pendenza_target=pendenza_target
)


# ===============================
# 2. Traiettoria sciatore
# ===============================
t = np.linspace(0.0, L, 300)      # parametro lungo la pista
y_traj = t
x_traj = 10.0 * np.sin(t / 2.0)
z_traj = h(x_traj, y_traj)

# Calcolo s (ascissa curvilinea)
dx_ds_approx = np.gradient(x_traj)
dy_ds_approx = np.gradient(y_traj)
dz_ds_approx = np.gradient(z_traj)
ds = np.sqrt(dx_ds_approx**2 + dy_ds_approx**2 + dz_ds_approx**2)
s = np.cumsum(ds)
s -= s[0]  # facciamo partire da 0

# ===============================
# 3. Calcolo angoli alpha (pendenza) e beta (sterzata)
# ===============================
delta = 1e-4
N = len(s)

alpha_deg = np.zeros(N)
beta_deg = np.zeros(N)

dx = np.gradient(x_traj)
dy = np.gradient(y_traj)

for i in range(N):
    xi = x_traj[i]
    yi = y_traj[i]

    # Gradiente numerico della superficie
    hx = (h(np.array([xi + delta]), np.array([yi])) -
          h(np.array([xi - delta]), np.array([yi]))) / (2.0 * delta)
    hy = (h(np.array([xi]), np.array([yi + delta])) -
          h(np.array([xi]), np.array([yi - delta]))) / (2.0 * delta)

    hx = hx.item()
    hy = hy.item()
    print(hx, hy)

    # Direzione di moto unitizzata
    vx, vy = dx[i], dy[i]
    norm_v = np.hypot(vx, vy)
    ux, uy = vx / norm_v, vy / norm_v

    # Derivata della quota lungo la direzione di moto
    #dh_ds = hx * ux + hy * uy    # QUI compare il segno! fondamentale -> PER ORA NON FACCIAMO LA DERIVATA NELLA DIR DI MODO MA LUNGO LA DIREZIONE Y 

    # Angolo di pendenza con segno
    alpha_rad = -np.arctan(hy)
    alpha_deg[i] = np.degrees(alpha_rad)

    # Beta: angolo tra direzione di moto e linea di massima pendenza
    vec_v  = np.array([dx[i], dy[i]])
    vec_ref = np.array([0.0, 1.0])

    num = np.dot(vec_v, vec_ref)
    den = np.linalg.norm(vec_v) * np.linalg.norm(vec_ref)

    if den < 1e-9:
        beta_mag_deg = 0.0
    else:
        cosang = np.clip(num / den, -1.0, 1.0)
        beta_mag_deg = np.degrees(np.arccos(cosang))   # modulo dell'angolo

    # SEGNO: vec_ref = (0, -1) ⇒ sign(beta) = -sign(vx)
    sign_beta = np.sign(dx[i])   # dx[i] è vx

    beta_deg[i] = sign_beta * beta_mag_deg       #BETA ANGOLO RISPETTO ALLA VERTICALE !!!!


# Convertiamo subito in radianti e costruiamo interpolanti in s
alpha_rad_arr = np.radians(alpha_deg)
beta_rad_arr = np.radians(beta_deg)

alpha_of_s = interp1d(s, alpha_rad_arr, kind='linear', fill_value="extrapolate")
beta_of_s  = interp1d(s, beta_rad_arr,  kind='linear', fill_value="extrapolate")

# ===============================
# Calcolo raggio di curvatura R(s)
# ===============================
P = np.column_stack([x_traj, y_traj])  # punti (x, y)
Npts = len(P)

R_vals = np.full(Npts, np.inf)  # ai bordi mettiamo infinito (rettilineo)

for i in range(1, Npts - 1):
    p_prev = P[i - 1]
    p      = P[i]
    p_next = P[i + 1]

    # Lati del triangolo
    a = np.linalg.norm(p_prev - p)
    b = np.linalg.norm(p - p_next)
    c = np.linalg.norm(p_next - p_prev)

    # Area del triangolo (formula di Erone)
    s_semi = 0.5 * (a + b + c)
    area_sq = s_semi * (s_semi - a) * (s_semi - b) * (s_semi - c)

    if area_sq <= 1e-16:
        R_vals[i] = np.inf
    else:
        area = np.sqrt(area_sq)
        curvature_i = 4.0 * area / (a * b * c)
        R_vals[i] = 1.0 / curvature_i

# Interpolante R(s)
def make_safe_interp1d(x, y, kind="linear"):
    x = np.asarray(x)
    y = np.asarray(y)

    # 1. Ordino per sicurezza
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # 2. Tolgo eventuali NaN/inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    x_min = float(x[0])
    x_max = float(x[-1])

    base_interp = interp1d(
        x,
        y,
        kind=kind,
        bounds_error=False,             # niente errori fuori range
        fill_value=(y[0], y[-1])        # fuori range → blocco al bordo
    )

    def safe_f(x_new):
        x_new = np.asarray(x_new)
        x_clip = np.clip(x_new, x_min, x_max)  # CLIP nel range [x_min, x_max]
        return base_interp(x_clip)

    # salvo i limiti per debug
    safe_f.x_min = x_min
    safe_f.x_max = x_max

    return safe_f

#R_of_s = interp1d(s, R_vals, kind='linear', fill_value='extrapolate')
R_of_s = make_safe_interp1d(s, R_vals, kind="linear")



# ===============================
# 4. Parametri fisici
# ===============================
m   = 80.0
g   = 9.81
mu  = 0.05
rho = 1.225
CdA = 0.3

# ===============================
# 5. Definizione dell'equazione differenziale in s
#    Stato = w = v^2
# ===============================

def ode_w(s_val, w):
    """
    ODE: dw/ds = (2/m) * F_net(s, v), con v = sqrt(w)
    s_val: ascissa curvilinea (scalare)
    w: array di lunghezza 1 (v^2)
    """
    w = w[0]
    if w < 0:
        v = 0.0
    else:
        v = np.sqrt(w)

    # Se v ~ 0, la dinamica si ferma (niente tempo evolutivo interessante)
    # ma possiamo comunque definire F_net
    alpha = alpha_of_s(s_val)   # pendenza terreno
    beta  = beta_of_s(s_val)    # sterzata rispetto alla fall line

    # Forza peso
    Fg = m * g

    # Raggio di curvatura nel punto corrente
    R_local = R_of_s(s_val)

    # Forza centrifuga (in realtà centripeta, ma appare così nel riferimento dello sciatore)
    side = np.sign(beta)  # +1 curva da una parte, -1 curva dall’altra

    if np.isinf(R_local) or (R_local == 0):
        F_centrifuga = 0.0
    else:
        F_centrifuga = side * m * v * v / abs(R_local)


    # Normale totale (gravità + curvatura)
    Fn = Fg * np.cos(alpha) 


    # Componente lungo massima pendenza
    Fs = Fg * np.sin(alpha) #OK

    # Proiezione lungo la traiettoria
    #Attenzione: l'angolo beta non è lo stesso scritto sui fogli
    F_p = Fs * np.cos(beta)#OK

    #F_lat
    F_lat = Fs * np.sin(beta)

    # Drag aerodinamico
    F_drag = 0.5 * rho * CdA * v**2 #OK

    #F_load 
    F_load = np.sqrt((Fn)**2+ (F_centrifuga+F_lat)**2)

    # Attrito neve
    F_fric = mu * F_load #invece che f_n alla fine voglio usare F_load

    # Forza netta lungo la traiettoria
    F_net = F_p - F_drag - F_fric #OK

    # ODE in w = v^2
    dw_ds = 2.0 * F_net / m
    return [dw_ds]

def stop_event(s_val, w):
    # vogliamo fermare l’integrazione quando w = v^2 arriva a 0
    return w[0]

stop_event.terminal = True      # interrompe l’integrazione
stop_event.direction = -1       # ci interessa quando w scende verso 0

# ===============================
# 6. Integrazione con solve_ivp
# ===============================
v0 = 5.0      # condizione iniziale (m/s)
w0 = v0**2

s_span = (float(s[0]), float(s[-1]))

sol = solve_ivp(
    ode_w,
    s_span,
    [w0],
    events=stop_event,
    t_eval=s,          # valutiamo esattamente agli stessi punti s
    method='RK45',     # va benissimo per iniziare
    rtol=1e-6,
    atol=1e-9
)

s_sol = sol.t          # s effettivo fino allo stop
w_sol = sol.y[0]
w_sol = np.maximum(w_sol, 0.0)
v_sol = np.sqrt(w_sol)


# ===============================
# 7. Calcolo tempo a posteriori
# ===============================
time = np.zeros_like(s_sol)
for i in range(len(s_sol) - 1):
    ds_step = s_sol[i+1] - s_sol[i]
    v_avg = 0.5 * (v_sol[i] + v_sol[i+1])
    if v_avg > 1e-3:
        dt = ds_step / v_avg
    else:
        dt = 0.0
    time[i+1] = time[i] + dt

# ===============================
# 8. Quantità derivate (una volta sola)
# ===============================

# Accelerazione tangenziale: a_t = v dv/ds
dv_ds = np.gradient(v_sol, s_sol)
a_t = v_sol * dv_ds

# Forza netta lungo traiettoria
F_net_vec = np.zeros_like(s_sol)
for i, s_val in enumerate(s_sol):
    w_tmp = w_sol[i]
    v_tmp = np.sqrt(max(w_tmp, 0.0))
    alpha = alpha_of_s(s_val)
    beta  = beta_of_s(s_val)

    Fg = m * g
    Fn = Fg * np.cos(alpha)
    Fs = Fg * np.sin(alpha)
    F_p = Fs * np.cos(beta)
    F_drag = 0.5 * rho * CdA * v_tmp**2
    F_fric = mu * Fn
    F_net_vec[i] = F_p - F_drag - F_fric

# Alpha e Beta sui punti s_sol
alpha_rad_sol = alpha_of_s(s_sol)
beta_rad_sol  = beta_of_s(s_sol)
alpha_deg_sol = np.degrees(alpha_rad_sol)
beta_deg_sol  = np.degrees(beta_rad_sol)

# Traiettoria effettivamente percorsa (ricostruita da s_sol)
x_eff = np.interp(s_sol, s, x_traj)
y_eff = np.interp(s_sol, s, y_traj)
z_eff = np.interp(s_sol, s, z_traj)

# Energia meccanica: E = T + V = 1/2 m v^2 + m g z
E_mech = 0.5 * m * v_sol**2 + m * g * z_eff

# ===============================
# 9. Plot – finestre separate
# ===============================

# ---- 1) Velocità ----
plt.figure(figsize=(6, 4))
plt.plot(s_sol, v_sol * 3.6, lw=2)
plt.title('Profilo Velocità (integrazione in s)')
plt.xlabel('s [m]')
plt.ylabel('Velocità [km/h]')
plt.grid(True)

# ---- 2) Accelerazione tangenziale ----
plt.figure(figsize=(6, 4))
plt.plot(s_sol, a_t, 'r', lw=1.5)
plt.title('Accelerazione tangenziale')
plt.xlabel('s [m]')
plt.ylabel('a_t [m/s²]')
plt.grid(True)
plt.axhline(0, color='k', linestyle='--')

# ---- 3) Forza netta lungo traiettoria ----
plt.figure(figsize=(6, 4))
plt.plot(s_sol, F_net_vec, 'k')
plt.title('Forza netta lungo traiettoria')
plt.xlabel('s [m]')
plt.ylabel('F_net [N]')
plt.grid(True)

# ---- 4) Pendenza (Alpha) ----
plt.figure(figsize=(6, 4))
plt.plot(s_sol, alpha_deg_sol, 'b-', lw=2)
plt.axhline(np.degrees(np.arctan(pendenza_target)),
            color='r', linestyle='--', label='Teorico piano')
plt.title('Alpha: pendenza terreno lungo traiettoria')
plt.xlabel('s [m]')
plt.ylabel('Alpha [°]')
plt.ylim([-20, 20])
plt.grid(True)
plt.legend()

# ---- 5) Angolo Beta ----
plt.figure(figsize=(6, 4))
plt.plot(s_sol, beta_deg_sol, 'm-', lw=2)
plt.title('Beta: angolo rispetto alla massima pendenza')
plt.xlabel('s [m]')
plt.ylabel('Beta [°]')
plt.grid(True)
plt.axhline(90, color='k', linestyle='--')

# ---- 6) Energia meccanica ----
plt.figure(figsize=(6, 4))
plt.plot(s_sol, E_mech, lw=2)
plt.title('Energia meccanica lungo la traiettoria')
plt.xlabel('s [m]')
plt.ylabel('E_mech [J]')
plt.grid(True)

# ---- 7) Superficie + traiettoria ----
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, Zg, alpha=0.4, edgecolor='none')

# Traiettoria teorica (completa)
ax.plot(x_traj, y_traj, z_traj, 'r', lw=2, label="Traiettoria definita")

# Traiettoria realmente percorsa
ax.plot(x_eff, y_eff, z_eff, 'g', lw=3, label="Traiettoria percorsa")

# Punto di arresto
ax.scatter(x_eff[-1], y_eff[-1], z_eff[-1], c='k', s=50, label='Stop')

ax.set_title('Superficie + Traiettoria percorsa')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.invert_xaxis()

ax.legend()
plt.tight_layout()
plt.show()






