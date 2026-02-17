import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import h5py


import warnings
warnings.filterwarnings("error")

#==========================================
# TRAIETTORIA SCIATORE
#==========================================
hdf5_file = 'segment_81222.h5' 
with h5py.File(hdf5_file, 'r') as data:
    pos = data['processed/com/pos'][:, :]

pos = pos[1200:]
x_traj = pos[:, 0]
y_traj = pos[:, 1]
z_traj = pos[:, 2]

#Calcolo ascissa curvilinea
dx_dt = np.gradient(x_traj)
dy_dt = np.gradient(y_traj)
dz_dt = np.gradient(z_traj)
ds = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)
s = np.cumsum(ds)

#===========================================
# CALCOLO DI ALPHA E BETA
#===========================================
#ALPHA
#minore di 0 per salita
#maggiore di 0 per discesa
N = len(x_traj)

alpha_deg = np.zeros(N)

x_traj = np.round(x_traj, 2)
y_traj = np.round(y_traj, 2)
z_traj = np.round(z_traj, 2)


for i in range(1, N-2):
    dx = x_traj[i+1] - x_traj[i]
    dy = y_traj[i+1] - y_traj[i]
    dz = z_traj[i+1] - z_traj[i]

    alpha = -np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    alpha_deg[i] = np.degrees(alpha)

# Riempimento bordi
alpha_deg[N-1] = alpha_deg[N-2]

# BETA -> angolo rispetto alla verticale
# calcolo vettore di riferimento (d) e angolo gamma 
dx_ref = x_traj[-1] - x_traj[0]
dy_ref = y_traj[-1] - y_traj[0]
dz_ref = z_traj[-1] - z_traj[0]
vec_ref = np.array([dx_ref, dy_ref, dz_ref])

gamma = -np.arctan2(dz_ref, np.sqrt(dx_ref**2 + dy_ref**2))
gamma_deg = np.degrees(gamma)
factor = np.cos(np.radians(gamma_deg - alpha_deg))   # (N,)
vec_ref = factor[:, None] * vec_ref                  # (N,3)

vec_ref_norm = np.linalg.norm(vec_ref, axis=1)
vec_ref_hat = vec_ref / vec_ref_norm[:, None]


#calcolo vettore tangente
T = np.column_stack([dx_dt, dy_dt, dz_dt])
T_norm = np.linalg.norm(T, axis=1)
T_hat = T / T_norm[:, None] 

#Angolo tra i due vettori
cosang = np.sum(vec_ref_hat * T_hat, axis=1)         # (N,)
cosang = np.clip(cosang, -1.0, 1.0)
beta_deg = np.degrees(np.arccos(cosang))             # (N,)
# Il segno di beta è dato da sign(T_hat x vec_ref_hat)
cross_beta = T_hat[:,0]*vec_ref_hat[:,1] - T_hat[:,1]*vec_ref_hat[:,0]
beta_sign = np.sign(cross_beta)

# Conversione in radianti e costruzioni interpolanti
alpha_rad_arr = np.radians(alpha_deg)
beta_rad_arr = np.radians(beta_sign*beta_deg)# ora beta ha il segno giusto per dare il segno alla forza laterale

alpha_of_s = interp1d(s, alpha_rad_arr, kind='linear', fill_value="extrapolate")
beta_of_s  = interp1d(s, beta_rad_arr,  kind='linear', fill_value="extrapolate")

#==========================================
# CALCOLO RAGGIO DI CURVATURA R(s)
#==========================================
P = np.column_stack([x_traj, y_traj, z_traj])
R_vals = np.full(N, np.inf)

k = 50

for i in range(k, N-k-1):
    p_prev = P[i-k]
    p      = P[i]
    p_next = P[i+k]

    v1 = p - p_prev
    v2 = p_next - p

    a = np.linalg.norm(p_prev - p)
    b = np.linalg.norm(p - p_next)
    c = np.linalg.norm(p_next - p_prev)

    cross = (v1[0]*v2[1] - v1[1]*v2[0])
    #Cambio il segno al cross product per fare in modo che il segno
    # sia giusto rispetto al sistema di riferimento locale scelto
    area = 0.5 * abs(cross)

    if area <= 1e-16:
        R_vals[i] = np.inf
    else:
        curvature = 4.0 * area / (a * b * c)
        sign = np.sign(cross)   
        curvature *= sign
        R_vals[i] = 1.0 / curvature

first_valid = k
last_valid  = N - k - 1

R_vals[:first_valid] = R_vals[first_valid]
R_vals[last_valid+1:] = R_vals[last_valid]

R_of_s = interp1d(s, R_vals, kind="linear")

# abbiamo fatto delle interpolanti per alpha, beta e R così i punti su cui risolviamo 
# l'ode possono essere anche di più di quelli che abbiamo per calcorale gli 
# angoli e il raggio di curvatura
#==========================================
# PARAMETRI FISICI
#==========================================
m = 80
g = 9.81
mu = 0.17
rho = 1.225
CdA = 0.3

#==========================================
# DEFINIZIONE ODE in s
#==========================================

def func_ode(s_val, w):
    """
    Viene fatto un cambio di variabile per togliere la v dal denominatore
    ODE: dw/ds = (2/m) * F_par(s, v), con v = sqrt(w)
    s_val: ascissa curvilinea (scalare)
    w: array di lunghezza 1 (v^2)
    """
    w = w[0]
    v = np.sqrt(w)

    alpha = alpha_of_s(s_val) #pendenza terreno
    beta = beta_of_s(s_val)   #angolo rispetto alla fall line
    R = R_of_s(s_val)

    #DEFINIZIONE FORZE
    #Forza peso
    Fg = m * g

    #Scomposizione in Fn e Fs e poi F_p e F_lat
    Fn = Fg * np.cos(alpha)
    Fs = Fg * np.sin(alpha)

    F_p = Fs * np.cos(np.abs(beta))
    F_lat = Fs * np.sin(np.abs(beta))

    #Forza centrifuga
    if np.isinf(R):
        F_centrifuga = 0.0
    else:
        F_centrifuga = m*v*v/R

    #Forza di drag
    F_drag = 0.5 *rho* CdA * v * v

    #Forza load
    F_load = np.sqrt((Fn)**2+ (F_centrifuga+np.sign(beta)*F_lat)**2)

    #Attrito neve
    F_fric = mu * F_load

    #Forza netta lungo la traiettoria 
    F_net = F_p - F_drag - F_fric

    #ODE
    dw_ds = 2.0*F_net / m
    return[dw_ds]


# Fermiamo l'intregrazione quando w = v^2 passa da positivo a negativo
# Dall'evoluzione dell'eq. differenziale w può diventare negativo ma 
# non ha significato fisico perchè w= v^2 quindi mi devo fermare
def stop_event(s_val, w):
    return w[0]   

stop_event.terminal = True  # interrompe l’integrazione
stop_event.direction = -1   # ci interessa quando w scende verso 0

# =========================================
# INTEGRAZIONE
# =========================================
v0 = 7.4      # condizione iniziale (m/s)
w0 = v0**2

s_span = (float(s[0]), float(s[-1]))

sol = solve_ivp(
    func_ode,
    s_span,
    [w0],
    events=stop_event,
    t_eval=s,          
    method='RK45',    
    rtol=1e-6,
    atol=1e-9
)

s_sol = sol.t          # s effettivo fino allo stop
w_sol = sol.y[0]
w_sol = np.maximum(w_sol, 0.0)  # Per stare sicuri prima di fare la radice
v_sol = np.sqrt(w_sol)

# =========================================
# CALCOLO TEMPI A POSTERIORI
# =========================================
time = np.zeros_like(s_sol)
for i in range(len(s_sol) - 1):
    ds_step = s_sol[i+1] - s_sol[i]
    v_avg = 0.5 * (v_sol[i] + v_sol[i+1])
    if v_avg > 1e-3:
        dt = ds_step / v_avg
    else:
        dt = 0.0
    time[i+1] = time[i] + dt

# =========================================
# CALCOLO FORZE
# =========================================
# Accelerazione tangenziale: a_t = v dv/ds
dv_ds = np.gradient(v_sol, s_sol)
a_t = v_sol * dv_ds

# Vettori per salvare le forze 
F_p_vec        = np.zeros_like(s_sol)
F_lat_vec      = np.zeros_like(s_sol)
F_cent_vec     = np.zeros_like(s_sol)
F_load_vec     = np.zeros_like(s_sol)
F_drag_vec     = np.zeros_like(s_sol)
F_fric_vec     = np.zeros_like(s_sol)
F_net_vec      = np.zeros_like(s_sol)

for i, s_val in enumerate(s_sol):
    v_tmp = v_sol[i]

    alpha = alpha_of_s(s_val)   # [rad]
    beta  = beta_of_s(s_val)    # [rad]
    R_local = R_of_s(s_val)

    # Peso
    Fg = m * g 
    Fn = Fg * np.cos(alpha)
    Fs = Fg * np.sin(alpha)
    F_p = Fs * np.cos(np.abs(beta))
    F_lat = Fs * np.sin(np.abs(beta))

    if np.isinf(R_local):
        F_centrifuga = 0.0
    else:
        F_centrifuga = m * v_tmp * v_tmp / R_local
    
    F_lat_tot = F_centrifuga+np.sign(beta)*F_lat
    F_load = np.sqrt(Fn**2 + F_lat_tot**2)
    F_drag = 0.5 * rho * CdA * v_tmp**2
    F_fric = mu * F_load
    F_net = F_p - F_drag - F_fric

    # salvo le forze
    F_p_vec[i]    = F_p
    F_lat_vec[i]  = F_lat
    F_cent_vec[i] = F_centrifuga
    F_load_vec[i] = F_load
    F_drag_vec[i] = F_drag
    F_fric_vec[i] = F_fric
    F_net_vec[i]  = F_net

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

#==========================================
# GRAFICI
#==========================================

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
#plt.axhline(np.degrees(np.arctan(pendenza_target)),
#            color='r', linestyle='--', label='Teorico piano')
plt.title('Alpha: pendenza terreno lungo traiettoria')
plt.xlabel('s [m]')
plt.ylabel('Alpha [°]')
plt.ylim([-20, 20])
plt.grid(True)
#plt.legend()

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

# ---- 6) Forza laterale ----
plt.figure(figsize=(6, 4))
plt.plot(s_sol, F_lat_vec, 'b-', lw=2)
plt.title('Forza laterale F_lat')
plt.xlabel('s [m]')
plt.ylabel('F_lat [N]')
plt.grid(True)

# ---- 7) Forza centrifuga ----
plt.figure(figsize=(6, 4))
plt.plot(s_sol, F_cent_vec, 'r-', lw=2)
plt.title('Forza centrifuga F_centr')
plt.xlabel('s [m]')
plt.ylabel('F_centr [N]')
plt.grid(True)



# ---- 7) Superficie + traiettoria ----
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(Xg, Yg, Zg, alpha=0.4, edgecolor='none')


#RAGGIO DI CURVATURA
R_max = 100   # scegli una soglia adatta al tuo problema
mask = np.isfinite(R_vals) & (np.abs(R_vals) < R_max)

plt.figure(figsize=(7,4))
plt.plot(s[mask], R_vals[mask], lw=2)
plt.xlabel('Ascissa curvilinea s')
plt.ylabel('Raggio di curvatura R')
plt.title('R(s) - raggio di curvatura')
plt.grid(True)

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


# #TRAIETTORIA
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_traj, y_traj, z_traj, linewidth=1)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.set_title("3D COM trajectory")

# #ALPHA-index
# plt.figure()
# plt.plot(alpha_deg)
# plt.xlabel("Step index")
# plt.ylabel("Alpha [deg]")
# plt.title("Slope angle along trajectory (index)")
# plt.grid(True)


# # BETA-index
# plt.figure()
# plt.plot(beta_deg[100:])
# plt.xlabel("Step index")
# plt.ylabel("Beta [deg]")
# plt.title('Angolo tra T_hat e vec_ref_hat (index)')
# plt.grid(True)

# #ALPHA E BETA interp
# alpha_rad = alpha_of_s(s)
# beta_rad  = beta_of_s(s)
# alpha_deg = np.degrees(alpha_rad)
# beta_deg  = np.degrees(beta_rad )

# plt.figure(figsize=(6, 4))
# plt.plot(s, alpha_deg, 'b-', lw=2)
# plt.title('Alpha: pendenza terreno lungo traiettoria (s)')
# plt.xlabel('s [m]')
# plt.ylabel('Alpha [°]')
# plt.ylim([-20, 20])
# plt.grid(True)

# plt.figure(figsize=(6, 4))
# plt.plot(s, beta_deg, 'm-', lw=2)
# plt.title('Beta: angolo rispetto alla massima pendenza (s)')
# plt.xlabel('s [m]')
# plt.ylabel('Beta [°]')
# plt.grid(True)
# plt.axhline(90, color='k', linestyle='--')

# #RAGGIO DI CURVATURA
# R_max = 1e2   # metri, oppure quello che ha senso nel tuo modello
# mask = np.isfinite(R_vals) & (np.abs(R_vals) < R_max)

# plt.figure(figsize=(7,4))
# plt.plot(s[mask], R_vals[mask], lw=2)
# plt.xlabel('Ascissa curvilinea s')
# plt.ylabel('Raggio di curvatura R')
# plt.title('R(s) - raggio di curvatura')
# plt.grid(True)


# #TRAIETTORIA+TANGENTE+DIR_RIF
# fig = plt.figure(figsize=(9,7))
# ax = fig.add_subplot(111, projection='3d')

# # Traiettoria
# ax.plot(x_traj, y_traj, z_traj, color='black', linewidth=2, label='Traiettoria')

# # Campioniamo per non avere troppe frecce
# step = 200   # modifica se vuoi più/meno frecce

# # Vettore tangente (blu)
# ax.quiver(
#     x_traj[::step],
#     y_traj[::step],
#     z_traj[::step],
#     T_hat[::step, 0],
#     T_hat[::step, 1],
#     T_hat[::step, 2],
#     length=10,
#     color='blue',
#     normalize=False
# )

# # Vettore di riferimento (rosso)
# ax.quiver(
#     x_traj[::step],
#     y_traj[::step],
#     z_traj[::step],
#     vec_ref_hat[::step, 0],
#     vec_ref_hat[::step, 1],
#     vec_ref_hat[::step, 2],
#     length=10,
#     color='red',
#     normalize=False
# )

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Traiettoria con T_hat (blu) e vec_ref_hat (rosso)')
# ax.set_box_aspect([1,1,1])   # scala uniforme

# plt.show()