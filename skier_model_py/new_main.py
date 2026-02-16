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

pos = pos[200:]
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

# Conversione in radianti e costruzioni interpolanti
alpha_rad_arr = np.radians(alpha_deg)
beta_rad_arr = np.radians(beta_deg)

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

    cross = v1[0]*v2[1] - v1[1]*v2[0]
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

#==========================================
# GRAFICI
#==========================================
#TRAIETTORIA
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_traj, y_traj, z_traj, linewidth=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D COM trajectory")

#ALPHA-index
plt.figure()
plt.plot(alpha_deg)
plt.xlabel("Step index")
plt.ylabel("Alpha [deg]")
plt.title("Slope angle along trajectory (index)")
plt.grid(True)


# BETA-index
plt.figure()
plt.plot(beta_deg[100:])
plt.xlabel("Step index")
plt.ylabel("Beta [deg]")
plt.title('Angolo tra T_hat e vec_ref_hat (index)')
plt.grid(True)

#ALPHA E BETA interp
alpha_rad = alpha_of_s(s)
beta_rad  = beta_of_s(s)
alpha_deg = np.degrees(alpha_rad)
beta_deg  = np.degrees(beta_rad )

plt.figure(figsize=(6, 4))
plt.plot(s, alpha_deg, 'b-', lw=2)
plt.title('Alpha: pendenza terreno lungo traiettoria (s)')
plt.xlabel('s [m]')
plt.ylabel('Alpha [°]')
plt.ylim([-20, 20])
plt.grid(True)

plt.figure(figsize=(6, 4))
plt.plot(s, beta_deg, 'm-', lw=2)
plt.title('Beta: angolo rispetto alla massima pendenza (s)')
plt.xlabel('s [m]')
plt.ylabel('Beta [°]')
plt.grid(True)
plt.axhline(90, color='k', linestyle='--')

#RAGGIO DI CURVATURA
R_max = 1e2   # metri, oppure quello che ha senso nel tuo modello
mask = np.isfinite(R_vals) & (np.abs(R_vals) < R_max)

plt.figure(figsize=(7,4))
plt.plot(s[mask], R_vals[mask], lw=2)
plt.xlabel('Ascissa curvilinea s')
plt.ylabel('Raggio di curvatura R')
plt.title('R(s) - raggio di curvatura')
plt.grid(True)


#TRAIETTORIA+TANGENTE+DIR_RIF
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

# Traiettoria
ax.plot(x_traj, y_traj, z_traj, color='black', linewidth=2, label='Traiettoria')

# Campioniamo per non avere troppe frecce
step = 200   # modifica se vuoi più/meno frecce

# Vettore tangente (blu)
ax.quiver(
    x_traj[::step],
    y_traj[::step],
    z_traj[::step],
    T_hat[::step, 0],
    T_hat[::step, 1],
    T_hat[::step, 2],
    length=10,
    color='blue',
    normalize=False
)

# Vettore di riferimento (rosso)
ax.quiver(
    x_traj[::step],
    y_traj[::step],
    z_traj[::step],
    vec_ref_hat[::step, 0],
    vec_ref_hat[::step, 1],
    vec_ref_hat[::step, 2],
    length=10,
    color='red',
    normalize=False
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Traiettoria con T_hat (blu) e vec_ref_hat (rosso)')
ax.set_box_aspect([1,1,1])   # scala uniforme

plt.show()