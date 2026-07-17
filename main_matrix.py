import argparse, random, os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tqdm import tqdm

from trajectory.trajectoryFromPorte import trajectoryLoader

from skier_model_py.physical_model import esegui_simulazione
from skier_model_py.fall_simulation import simula_caduta
from trajectory.tangent_derivative import obtain_inclination, obtain_slope_borders, sample_surface_z

from tqdm.contrib.concurrent import process_map
from scipy.ndimage import convolve

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
    der_alfa = np.gradient(alfa)

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
    count_dalfa_zeros = (c[np.minimum(j + sw, N)] - c[np.minimum(np.maximum(j - sw, 0) + 1, N)] + mask[np.maximum(j - sw, 0)]).astype(float)    
    
    
    haz_coeff = count_dalfa_zeros+count_dgamma_zeros

    haz_coeff = (haz_coeff-np.min(haz_coeff))/(np.max(haz_coeff)-np.min(haz_coeff) + 1e-9)

    return haz_coeff

def set_seed(seed:int) -> int:
    np.random.seed(seed)
    random.seed(seed)
    return seed

def return_inclination(x_utm:np.ndarray, y_utm:np.ndarray, z_traj:np.ndarray, path_to_las:str):    
    alpha_deg, grads, real_z = obtain_inclination(x_utm,y_utm,path_to_las)
    alpha_deg = -alpha_deg
    N = len(x_utm)
    
    if alpha_deg.shape[0] != N or np.isnan(alpha_deg).any():
        alpha_deg = np.zeros(N)
        for i in range(1, N-2):
            dx = x_utm[i+1] - x_utm[i]
            dy = y_utm[i+1] - y_utm[i]
            dz = z_traj[i+1] - z_traj[i]
            alpha_deg[i] = np.degrees(-np.arctan2(dz, np.sqrt(dx**2 + dy**2)))
    alpha_deg[N-1] = alpha_deg[N-2]
    alpha_deg[0] = alpha_deg[1] # Riempimento bordo iniziale

    return alpha_deg, grads, real_z
        
def _simulate(df, path_to_las:str):
    alphas, grads, real_z = return_inclination(df["Est [m]"].values, df["Nord [m]"].values, df["Quota Orto. [m]"].values, path_to_las)
    return esegui_simulazione(df["Est [m]"].values, df["Nord [m]"].values, df["Quota Orto. [m]"].values, alfa=alphas), alphas, grads, real_z

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
    

    maxX, maxY = max(df["Est [m]"].max()  for df in listDf), max(df["Nord [m]"].max() for df in listDf)
    minX, minY = min(df["Est [m]"].min()  for df in listDf), min(df["Nord [m]"].min() for df in listDf)

    risSim = process_map(
        _simulate,
        listDf,
        [args.path_to_las] * len(listDf),
        max_workers=os.cpu_count(),
        chunksize=1,
        desc="Simulating",
        unit="traj",
    )
    risSim, alphas, grads, real_z  = [res[0] for res in risSim], [res[1] for res in risSim], [res[2] for res in risSim], [res[3] for res in risSim]  # Extract the simulation results and the alphas

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

    haz_coeff_incl = [(alpha-np.min(alpha)) / (np.max(alpha)-np.min(alpha)) for alpha in alphas_der_sw]

    haz_coeff_incl = [(alpha-np.min(alpha)) / (np.max(alpha)-np.min(alpha)) for alpha in alphas]
    
    # for alpha in alphas_der:
    weight = 1/3
    haz_coeff = weight* np.array(haz_bump_coeffs) + weight * np.array(haz_coeff_sim) + weight * np.array(haz_coeff_incl)

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