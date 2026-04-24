import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore") # Nasconde i warning per divisioni per zero temporanee

def esegui_simulazione(x_traj, y_traj, z_traj, m=80, g=9.81, mu=0.16, rho=1.225, CdA=0.3, v0=4.91):
    """
    Calcola la fisica di uno sciatore lungo una traiettoria 3D fornita.
    
    Input:
    - x_traj, y_traj, z_traj: array delle coordinate (devono avere la stessa lunghezza)
    - ts_com: array dei timestamp associati a ogni punto
    - Parametri fisici opzionali (massa, gravità, attrito, densità aria, aerodinamica, vel. iniziale)
    
    Output:
    - Ritorna un dizionario con i profili di spazio, velocità e forze.
    """

    # Arrotondamento per eliminare rumore numerico
    x_traj = np.round(x_traj, 2)
    y_traj = np.round(y_traj, 2)
    z_traj = np.round(z_traj, 2)
    
    N = len(x_traj)
    
    # ==========================================
    # 1. CALCOLO ASCISSA CURVILINEA (s)
    # ==========================================
    # ==========================================
    # 1. CALCOLO ASCISSA CURVILINEA (s) - SENZA ts_com
    # ==========================================
    # Calcolo delle differenze spaziali tra punti consecutivi
    dx = np.diff(x_traj)
    dy = np.diff(y_traj)
    dz = np.diff(z_traj)

    # Distanza euclidea (teorema di Pitagora in 3D) per ogni segmento
    ds = np.sqrt(dx**2 + dy**2 + dz**2)

    # L'ascissa curvilinea è la somma cumulativa delle distanze
    s = np.concatenate(([0], np.cumsum(ds)))

    # ===========================================
    # 2. CALCOLO DI ALPHA E BETA
    # ===========================================
    alpha_deg = np.zeros(N)
    for i in range(1, N-2):
        dx = x_traj[i+1] - x_traj[i]
        dy = y_traj[i+1] - y_traj[i]
        dz = z_traj[i+1] - z_traj[i]
        alpha = -np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        alpha_deg[i] = np.degrees(alpha)

    alpha_deg[N-1] = alpha_deg[N-2]
    alpha_deg[0] = alpha_deg[1] # Riempimento bordo iniziale

    # BETA -> angolo rispetto alla verticale
    dx_ref = x_traj[-1] - x_traj[0]
    dy_ref = y_traj[-1] - y_traj[0]
    dz_ref = z_traj[-1] - z_traj[0]
    vec_ref = np.array([dx_ref, dy_ref, dz_ref])

    gamma = -np.arctan2(dz_ref, np.sqrt(dx_ref**2 + dy_ref**2))
    gamma_deg = np.degrees(gamma)
    factor = np.cos(np.radians(gamma_deg - alpha_deg))  
    vec_ref_arr = factor[:, None] * vec_ref                  

    vec_ref_norm = np.linalg.norm(vec_ref_arr, axis=1)
    # Evitiamo divisioni per zero
    vec_ref_norm[vec_ref_norm == 0] = 1e-10
    vec_ref_hat = vec_ref_arr / vec_ref_norm[:, None]

    # Calcolo del vettore tangente rispetto allo spazio (s) anziché al tempo
    dx_ds = np.gradient(x_traj, s)
    dy_ds = np.gradient(y_traj, s)
    dz_ds = np.gradient(z_traj, s)

    T = np.column_stack([dx_ds, dy_ds, dz_ds])
    T_norm = np.linalg.norm(T, axis=1)
    T_norm[T_norm == 0] = 1e-10
    T_hat = T / T_norm[:, None] 

    cosang = np.sum(vec_ref_hat * T_hat, axis=1)         
    cosang = np.clip(cosang, -1.0, 1.0)
    beta_deg = np.degrees(np.arccos(cosang))             
    
    cross_beta = T_hat[:,0]*vec_ref_hat[:,1] - T_hat[:,1]*vec_ref_hat[:,0]
    beta_sign = np.sign(cross_beta)

    alpha_rad_arr = np.radians(alpha_deg)
    beta_rad_arr = np.radians(beta_sign * beta_deg)

    # ==========================================
    # 3. CALCOLO RAGGIO DI CURVATURA R(s)
    # ==========================================
    P = np.column_stack([x_traj, y_traj, z_traj])
    R_vals = np.full(N, np.inf)

    # Usiamo un k adattivo nel caso la traiettoria abbia pochi punti
    k = min(50, max(1, N // 10)) 

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
        area = 0.5 * abs(cross)

        if area <= 1e-16 or a*b*c == 0:
            R_vals[i] = np.inf
        else:
            curvature = 4.0 * area / (a * b * c)
            sign = np.sign(cross)   
            curvature *= sign
            R_vals[i] = 1.0 / curvature

    R_vals[:k] = R_vals[k]
    R_vals[N-k-1:] = R_vals[N-k-2]

    # ==========================================
    # 4. CREAZIONE FUNZIONI INTERPOLANTI
    # ==========================================
    # Rimuoviamo eventuali duplicati in 's' per evitare errori in interp1d
    s_unique, indices = np.unique(s, return_index=True)
    
    alpha_of_s = interp1d(s_unique, alpha_rad_arr[indices], kind='linear', fill_value="extrapolate")
    beta_of_s  = interp1d(s_unique, beta_rad_arr[indices],  kind='linear', fill_value="extrapolate")
    R_of_s     = interp1d(s_unique, R_vals[indices], kind="linear", fill_value="extrapolate")

    # ==========================================
    # 5. DEFINIZIONE ODE E RISOLUZIONE
    # ==========================================
    def func_ode(s_val, w):
        w = max(w[0], 0.0)
        v = np.sqrt(w)

        alpha = alpha_of_s(s_val) 
        beta = beta_of_s(s_val)   
        R = R_of_s(s_val)

        Fg = m * g
        Fn = Fg * np.cos(alpha)
        Fs = Fg * np.sin(alpha)
        F_p = Fs * np.cos(np.abs(beta))
        F_lat = Fs * np.sin(np.abs(beta))

        if np.isinf(R):
            F_centrifuga = 0.0
        else:
            F_centrifuga = m*v*v/R

        F_drag = 0.5 * rho * CdA * v * v
        F_load = np.sqrt((Fn)**2 + (F_centrifuga + np.sign(beta)*F_lat)**2)
        F_fric = mu * F_load
        F_net = F_p - F_drag - F_fric

        dw_ds = 2.0 * F_net / m
        return [dw_ds]

    def stop_event(s_val, w):
        return w[0]   

    stop_event.terminal = True  
    stop_event.direction = -1   

    w0 = v0**2
    s_span = (float(s_unique[0]), float(s_unique[-1]))

    sol = solve_ivp(
        func_ode, s_span, [w0], events=stop_event, 
        t_eval=s_unique, method='RK45', rtol=1e-6, atol=1e-9
    )

    s_sol = sol.t
    w_sol = np.maximum(sol.y[0], 0.0)
    v_sol = np.sqrt(w_sol)

    # =========================================
    # 6. CALCOLO FORZE A POSTERIORI
    # =========================================
    F_lat_vec = np.zeros_like(s_sol)
    F_cent_vec = np.zeros_like(s_sol)
    F_net_vec = np.zeros_like(s_sol)
    R_vec = np.zeros_like(s_sol)

    for i, s_val in enumerate(s_sol):
        v_tmp = v_sol[i]
        alpha = alpha_of_s(s_val) 
        beta  = beta_of_s(s_val)  
        R_local = R_of_s(s_val)

        Fg = m * g 
        Fn = Fg * np.cos(alpha)
        Fs = Fg * np.sin(alpha)
        F_p = Fs * np.cos(np.abs(beta))
        F_lat = Fs * np.sin(np.abs(beta))

        if np.isinf(R_local):
            F_centrifuga = 0.0
        else:
            F_centrifuga = m * v_tmp * v_tmp / R_local
        
        F_lat_tot = F_centrifuga + np.sign(beta)*F_lat
        F_load = np.sqrt(Fn**2 + F_lat_tot**2)
        F_drag = 0.5 * rho * CdA * v_tmp**2
        F_fric = mu * F_load
        F_net = F_p - F_drag - F_fric

        F_lat_vec[i]  = F_lat_tot # Salviamo la forza laterale totale
        F_cent_vec[i] = F_centrifuga
        F_net_vec[i]  = F_net
        R_vec[i] = R_local

    # =========================================
    # 7. OUTPUT
    # =========================================
    # Restituiamo un dizionario in modo da poter accedere facilmente ai dati dal notebook
    return {
        's': s_sol,
        'v': v_sol,
        'F_lat': F_lat_vec,
        'F_cent': F_cent_vec,
        'F_net': F_net_vec,
        'R': R_vec,
    }