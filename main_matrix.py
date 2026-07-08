import argparse, random, os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from trajectory.trajectoryFromPorte import trajectoryLoader

from skier_model_py.physical_model import esegui_simulazione

from trajectory.tangent_derivative import obtain_inclination, obtain_slope_borders

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

    for j in range(sw, len(der_gamma)-sw):
        der_sw_gamma[j] = np.mean(der_gamma[j-sw:j+sw])
        
    for j in range(sw):
        der_sw_gamma[j] = np.mean(der_gamma[0:j+sw])
            
    for j in range(len(der_gamma)-sw, len(der_gamma)):
        der_sw_gamma[j] = np.mean(der_gamma[j-sw:len(der_gamma)])

    #COUNT NUMBER OF ZEROS IN SMOOTHED DERIVATIVE, in a sliding window
    count_dgamma_zeros = np.zeros_like(der_sw_gamma)
    for j in range(sw, len(der_gamma)-sw):        
        count_dgamma_zeros[j] += np.sum(np.diff(np.where(np.abs(der_sw_gamma[j-sw:j+sw]) < tol, 1, 0), prepend=0) == 1)
        
    for j in range(sw):
        count_dgamma_zeros[j] += np.sum(np.diff(np.where(np.abs(der_sw_gamma[0:j+sw]) < tol, 1, 0), prepend=0) == 1)
            
    for j in range(len(der_gamma)-sw, len(der_gamma)):
        count_dgamma_zeros[j] += np.sum(np.diff(np.where(np.abs(der_sw_gamma[j-sw:len(der_gamma)]) < tol, 1, 0), prepend=0) == 1)
    

    #Now do the same on alfa
    der_alfa = np.gradient(alfa)

    der_sw_alfa = np.zeros_like(der_alfa)

    for j in range(sw, len(der_alfa)-sw):
        der_sw_alfa[j] = np.mean(der_alfa[j-sw:j+sw])
        
    for j in range(sw):
        der_sw_alfa[j] = np.mean(der_alfa[0:j+sw])
            
    for j in range(len(der_alfa)-sw, len(der_alfa)):
        der_sw_alfa[j] = np.mean(der_alfa[j-sw:len(der_alfa)])
    
    count_dalfa_zeros = np.zeros_like(der_sw_alfa)
    for j in range(sw, len(der_alfa)-sw):        
        count_dalfa_zeros[j] += np.sum(np.diff(np.where(np.abs(der_sw_alfa[j-sw:j+sw]) < tol, 1, 0), prepend=0) == 1)
    
        
    for j in range(sw):
        count_dalfa_zeros[j] += np.sum(np.diff(np.where(np.abs(der_sw_alfa[0:j+sw]) < tol, 1, 0), prepend=0) == 1)
            
    for j in range(len(der_alfa)-sw, len(der_alfa)):
        count_dalfa_zeros[j] += np.sum(np.diff(np.where(np.abs(der_sw_alfa[j-sw:len(der_alfa)]) < tol, 1, 0), prepend=0) == 1)
    
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

    gates_path = args.gates
    print(f"Gates path: {gates_path}")

    # Load gates and generate new trajectories
    loader = trajectoryLoader(gates_path)
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
    weight = 0.5
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
    
    plt.figure(figsize=(10, 4))
    plt.plot(alphas[0], label="Inclination (degrees)")
    plt.xlabel("Point index")
    plt.ylabel("Inclination (degrees)")
    plt.title("Surface Inclination Along Trajectory")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(alphas_der[0], label="Inclination derivative (degrees)", linestyle='--')
    plt.xlabel("Point index")
    plt.ylabel("Inclination derivative (degrees)")
    plt.title("Surface Inclination Derivative Along Trajectory")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(alphas_der_sw[0], label="Inclination derivative (degrees)", linestyle='--')
    plt.xlabel("Point index")
    plt.ylabel("Inclination derivative (degrees)")
    plt.title("Surface Inclination Derivative Along Trajectory")
    plt.legend()
    plt.grid()
    plt.tight_layout()


    plt.figure(figsize=(10, 4))
    plt.plot(grads[0][:,0], label="Inclination w.r.t. X", linestyle='-')
    plt.plot(grads[0][:,1], label="Inclination w.r.t. Y", linestyle='-')
    plt.xlabel("Point index")
    plt.ylabel("Inclination (degrees)")
    plt.title("Surface Inclination Along Trajectory")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(np.degrees(np.arccos(grads[0][:,0])), label="Inclination w.r.t. X", linestyle='--')
    plt.plot(180-np.degrees(np.arccos(grads[0][:,1])), label="Inclination w.r.t. Y", linestyle='--')
    plt.xlabel("Point index")
    plt.ylabel("Inclination (degrees)")
    plt.title("Surface Inclination Along Trajectory")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    print(real_z[0])
    plt.figure(figsize=(10, 4))
    plt.plot(real_z[0], label="Inclination w.r.t. X", linestyle='-')
    plt.xlabel("Point index")
    plt.ylabel("Inclination (degrees)")
    plt.title("Surface Inclination Along Trajectory")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.show()

    haz_coeff_incl = [(alpha-np.min(alpha)) / (np.max(alpha)-np.min(alpha)) for alpha in alphas_der_sw]

    haz_coeff_incl = [(alpha-np.min(alpha)) / (np.max(alpha)-np.min(alpha)) for alpha in alphas]
    
    # for alpha in alphas_der:
    weight = 0.5
    haz_coeff = haz_bump_coeffs #weight * np.array(haz_coeff_sim) + (1-weight) * np.array(haz_coeff_incl)

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

    #CALCOLO TANGENTI VIE DI FUGA
    tangents = []
    for i in range(len(risSim)):
        len_traj = len_trajectories[i]
        min_p = -5*(num_points/len_traj)
        max_p = 2*(num_points/len_traj)
        haz_coeff = weight * haz_coeff_sim[i] + (1-weight) * haz_coeff_incl[i]
        lim_var = np.percentile(haz_coeff, 99)
        idxs = np.argwhere(haz_coeff >= lim_var)
        for max_idx in idxs:

            #max_idx = np.argmax(haz_coeff)
            for p in range(int(min_p), int(max_p)+1):
                idx = max(0, min(len(haz_coeff)-1, max_idx + p))
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
    # t = np.array([-20, 20])   # half-length of each drawn tangent, in meters
    # for nord, est, angle in tangents:
    #     est_idx  = (est  - minX) * sx
    #     nord_idx = (nord - minY) * sy
    #     # displayed image is riverMatrix.T: plot-x = Nord index, plot-y = Est index
    #     plt.plot(nord_idx + np.sin(angle) * sy * t, est_idx  + np.cos(angle) * sx * t, color='blue', linewidth=1.5)
    #     plt.plot(nord_idx, est_idx, marker='o', color='blue', markersize=4)


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
    

if __name__ == "__main__":
    SEED:int = 23
    NUM_POINTS:int = 3_000
    assert set_seed(SEED) == SEED, "Error setting the seed"
    main()