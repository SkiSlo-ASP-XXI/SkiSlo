import argparse
import os, sys

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from trajectory.trajectoryFromPorte import trajectoryLoader

from skier_model_py.physical_model import esegui_simulazione

def main(): #To call main paste: python main.py --gates data/pointsLocationFirstCourse.csv --numTrajectories 50
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
    
    for traj in tqdm(newTraj, desc="Processing trajectories", unit="trajectory", leave=False):
        trajectory = loader.prepareTrajectories(numPoints=3000, gates=traj)
        
        maxX = max(maxX, trajectory["Est [m]"].max())
        maxY = max(maxY, trajectory["Nord [m]"].max())
        minX = min(minX, trajectory["Est [m]"].min())
        minY = min(minY, trajectory["Nord [m]"].min())

        listDf.append(trajectory)
        risSim.append(esegui_simulazione(trajectory["Est [m]"].values, trajectory["Nord [m]"].values, trajectory["Quota Orto. [m]"].values))
    
    # Post processing of the results 
    haz_coeff = []
    for res in risSim:
        c_haz = np.abs(res['F_lat'])
        c_haz = c_haz / np.max(c_haz)  # Normalize to [0, 1] 
        haz_coeff.append(c_haz)

    # Create a river matrix of size 5000x5000 and fill it with the hazard coefficients (for visualization purposes)
    
    riverMatrix = np.zeros((int(maxX - minX)+1, int(maxY - minY)+1))
    reliabilityMatrix = np.zeros((int(maxX - minX)+1, int(maxY - minY)+1))
    # The matrix is such that the zero of the x-axis is the minimum x value of the trajectories and the zero of the y-axis is the minimum y value of the trajectories. 
    # The values of the matrix are filled with the hazard coefficients, where the position in the matrix corresponds to the position in the trajectory (x, y).
    for i in range(len(listDf)):
        x,y = listDf[i]["Est [m]"].values, listDf[i]["Nord [m]"].values
        x = ((x - minX) / (maxX - minX+1e-9) * (riverMatrix.shape[1]-1)).astype(int)  # Scale x to [0, maxShape-1]
        y = ((y - minY) / (maxY - minY+1e-9) * (riverMatrix.shape[0]-1)).astype(int)  # Scale y to [0, maxShape-1]
        riverMatrix[y, x] += haz_coeff[i]  # Fill the matrix with the hazard coefficient
        reliabilityMatrix[y, x] += 1
    
    riverMatrix2 = riverMatrix / (reliabilityMatrix + 1e-9)  # Average hazard coefficient for each position


    riverMatrix = 100 * riverMatrix/np.max(riverMatrix)
    reliabilityMatrix = 100 * reliabilityMatrix/np.max(reliabilityMatrix)
    


    # fig = plt.figure(figsize=(10,10))
    # plt.matshow(riverMatrix, cmap='hot', fignum=fig.number)
    print("Plotting the river matrix...")
    print(f"Max X: {maxX}, Min X: {minX}, Max Y: {maxY}, Min Y: {minY}")
    print(f"Difference x: {maxX - minX}, Difference y: {maxY - minY}")
    print("Non-zero values in river matrix:", len(riverMatrix[riverMatrix > 0]))
    print("Matrix shape :", riverMatrix.shape)


    cmap = plt.get_cmap('RdYlGn_r').copy()
    cmap.set_bad(color='white')

    plt.figure(figsize=(10, 10))
    plt.imshow(np.ma.masked_equal(riverMatrix, 0), cmap=cmap, origin='lower')
    plt.colorbar(label='Hazard coefficient')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.ma.masked_equal(reliabilityMatrix, 0), cmap=cmap, origin='lower')
    plt.colorbar(label='Reliability')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.ma.masked_equal(riverMatrix2, 0), cmap=cmap, origin='lower')
    plt.colorbar(label='Normalized Hazard coefficient')
    
    
    
    #show results
    plt.show()

    
if __name__ == "__main__": 
    main()