import argparse, os, random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from trajectory.trajectoryFromPorte import trajectoryLoader

from skier_model_py.physical_model import esegui_simulazione
import scipy.interpolate as scp


def set_seed(seed:int)->int:
    np.random.seed(seed)
    random.seed(seed)
    return seed




def main(): #To call main paste: python main_regression.py --gates data/pointsLocationFirstCourse.csv --numTrajectories 50
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
        trajectory = loader.prepareTrajectories(numPoints=1_000, gates=traj)
        
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

    total_coefficients = defaultdict(float)
    counter = defaultdict(float)
    for trajectory, coeff in tqdm(zip(listDf, haz_coeff), desc="Generating total risl", unit="trajectory", leave=False):
        for x,y,risk in zip(trajectory["Est [m]"].values, trajectory["Nord [m]"].values, coeff):            
            total_coefficients[(x,y)] += risk
            counter[(x,y)] += 1
    
    total_coefficients = {k: 100*total_coefficients[k]/(counter[k]) for k in total_coefficients.keys()}

    #FUNCTION FITTING
    xs = np.array([k[0] for k in total_coefficients.keys()])
    ys = np.array([k[1] for k in total_coefficients.keys()])
    z = np.array([ total_coefficients[(x,y)] for x,y in zip(xs, ys)])

    inter_param = scp.bisplrep(xs, ys, z,kx=4,ky=4)

    x = np.linspace(minX, maxX, 2_000)
    y = np.linspace(minY, maxY, 2_000)
    pred = scp.bisplev(x,y,inter_param)

    #plot pred
    plt.figure(figsize=(10, 8))
    plt.imshow(pred, extent=(minX, maxX, minY, maxY), origin='lower', cmap='viridis')
    plt.colorbar(label='Hazard Coefficient')
    plt.title('Interpolated Hazard Coefficient')
    plt.xlabel('Est [m]')
    plt.ylabel('Nord [m]')
    plt.savefig("interpolated_hazard_coefficient.png")
    plt.show()
    

    
if __name__ == "__main__": 
    SEED = 23
    assert set_seed(SEED) == SEED, "Error setting the seed"
    main()