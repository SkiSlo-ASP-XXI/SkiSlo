import argparse
from trajectory.trajectoryFromPorte import trajectoryLoader 
import os

from skier_model_py.SkierModel import SkierModel #TODO: add the file

def main():
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
    loader.saveNewTrajectories(newTraj, os.path.join(args.output, "simulated_trajectories.csv"))

    # If plotting is enabled, plot the original and simulated trajectories
    if args.plot:
        os.makedirs(args.plotPath, exist_ok=True)
        loader.plotSimulatedTrajectories(newTraj, args.plotPath)
    
    #create a skier model and simulate trajectories (TODO: add the file)
    skier_model = SkierModel()
    skier_model.simulate_trajectories(os.path.join(args.output, "simulated_trajectories")) #TODO: add the method to the SkierModel class,
    #simulate_trajectories must take in the path to the simulated trajectories and simulate them using the skier model, then save the results in a new folder called "simulations" inside the output folder

    

    





if __name__ == "__main__": main()