import argparse, random, os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from trajectory.trajectoryFromPorte import trajectoryLoader

from skier_model_py.physical_model import esegui_simulazione


from tqdm.contrib.concurrent import process_map
from scipy.ndimage import convolve


def set_seed(seed:int) -> int:
    np.random.seed(seed)
    random.seed(seed)
    return seed

def _simulate(df):
    return esegui_simulazione(df["Est [m]"].values, df["Nord [m]"].values, df["Quota Orto. [m]"].values)



def main(): #To call main paste: python main_matrix.py --gates data/pointsLocationFirstCourse.csv --numTrajectories 200
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
    
    # OLD CODE
    # for traj in tqdm(newTraj, desc="Processing trajectories", unit="trajectory", leave=False):
    #     trajectory = loader.prepareTrajectories(numPoints=3_000, gates=traj)
        
    #     maxX = max(maxX, trajectory["Est [m]"].max())
    #     maxY = max(maxY, trajectory["Nord [m]"].max())
    #     minX = min(minX, trajectory["Est [m]"].min())
    #     minY = min(minY, trajectory["Nord [m]"].min())

    #     listDf.append(trajectory)
    #     risSim.append(esegui_simulazione(trajectory["Est [m]"].values, trajectory["Nord [m]"].values, trajectory["Quota Orto. [m]"].values))
    

    listDf = [loader.prepareTrajectories(numPoints=3_000, gates=traj) for traj in newTraj]

    maxX, maxY = max(df["Est [m]"].max()  for df in listDf), max(df["Nord [m]"].max() for df in listDf)
    minX, minY = min(df["Est [m]"].min()  for df in listDf), min(df["Nord [m]"].min() for df in listDf)

    risSim = process_map(
        _simulate,
        listDf,
        max_workers=os.cpu_count(),
        chunksize=1,
        desc="Simulating",
        unit="traj",
    )

    # Post processing of the results 
    haz_coeff = [np.abs(res['F_lat']) / np.max(np.abs(res['F_lat'])) for res in risSim]

    # Create a river matrix with the cells of 1 meter x 1 meter and fill it with the hazard coefficients (for visualization purposes)
        
    riverMatrix = np.zeros((int(maxX - minX)+1, int(maxY - minY)+1))
    reliabilityMatrix = np.zeros((int(maxX - minX)+1, int(maxY - minY)+1))
    # The matrix is such that the zero of the x-axis is the minimum x value of the trajectories and the zero of the y-axis is the minimum y value of the trajectories. 
    # The values of the matrix are filled with the hazard coefficients, where the position in the matrix corresponds to the position in the trajectory (x, y).
    # 
    # OLD CODE
    #  for i in range(len(listDf)):
    #     x,y = listDf[i]["Est [m]"].values, listDf[i]["Nord [m]"].values
    #     x = ((x - minX) / (maxX - minX+1e-9) * (riverMatrix.shape[1]-1)).astype(int)  # Scale x to [0, maxShape-1]
    #     y = ((y - minY) / (maxY - minY+1e-9) * (riverMatrix.shape[0]-1)).astype(int)  # Scale y to [0, maxShape-1]
    #     riverMatrix[y, x] += haz_coeff[i]  # Fill the matrix with the hazard coefficient
    #     reliabilityMatrix[y, x] += 1
    
    # riverMatrix = riverMatrix / (reliabilityMatrix + 1e-9)  # Average hazard coefficient for each position

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


    # fig = plt.figure(figsize=(10,10))
    # plt.matshow(riverMatrix, cmap='hot', fignum=fig.number)
    print("Plotting the river matrix...")
    print(f"Max X: {maxX}, Min X: {minX}, Max Y: {maxY}, Min Y: {minY}")
    print(f"Difference x: {maxX - minX}, Difference y: {maxY - minY}")
    print("Non-zero values in river matrix:", len(riverMatrix[riverMatrix > 0]))
    print("Matrix shape :", riverMatrix.shape)


    cmap = plt.get_cmap('RdYlGn_r').copy()
    cmap.set_bad(color='white')

    plt.figure(figsize=(15, 15))
    plt.imshow(np.ma.masked_equal(riverMatrix.T, 0), cmap=cmap, origin='lower')
    plt.colorbar(label='Hazard coefficient')
    plt.savefig("river_matrix.pdf", bbox_inches='tight', format='pdf')



    plt.figure(figsize=(15, 15))
    plt.imshow(np.ma.masked_equal(reliabilityMatrix.T, 0), cmap=cmap, origin='lower')
    plt.colorbar(label='Reliability')
    plt.savefig("reliability_matrix.pdf", bbox_inches='tight', format='pdf')
    
    #show results
    plt.show()
    
if __name__ == "__main__":
    SEED = 23
    assert set_seed(SEED) == SEED, "Error setting the seed"
    main()