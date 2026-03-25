import json, os
import folium
import pandas as pd
import numpy as np
import argparse


from scipy.interpolate import make_splprep

class trajectoryLoader():
    VALID_NAMES = ['Est [m]', 'Nord [m]', 'Quota Orto. [m]', 'WGS84_Lat2', 'WGS84_Lon2']

    def __init__(self, path:str, picturesSavePath:str="./figures")->None:
        os.makedirs(picturesSavePath, exist_ok=True)
        self.__picturesSavePath = picturesSavePath
        self.__gates = pd.read_csv(path, sep=";", header=0, index_col=0)[self.VALID_NAMES] 

    @property
    def gates(self) -> pd.DataFrame:
        """Returns a copy of the gates DataFrame to prevent external modifications.
        
        Returns:
            portPositions (pd.DataFrame): A copy of the gates DataFrame.
        """
        return self.__gates.copy()

    # %% - PREPARATION -
    def __addFakeInitialAndFinalGates(self) -> None:
        """Adds fake initial and final gates to the trajectory based on the existing gates.
        The fake gates are created by extending the trajectory in both directions by a distance equal to the maximum distance between consecutive gates.
        
        This method modifies the internal gates DataFrame by adding two new rows: 'fakeInit' and 'fakeFinal', which represent the extended initial and final gates, respectively.
        """
        idx = self.__gates.columns.get_indexer(['WGS84_Lat2', 'WGS84_Lon2', 'Quota Orto. [m]'])

        distance_max_meter = max(map(lambda x: np.linalg.norm(x[0][idx] - x[1][idx]), zip(self.__gates.iloc[:-1].values, self.__gates.iloc[1:].values)))

        direction_vector = self.__gates.iloc[1, idx] - self.__gates.iloc[0, idx] # Direction from the first gate to the second gate
        direction_vector /= (np.linalg.norm(direction_vector) + 1e-8)  # Normalize the direction vector

        new_initial_gate, new_final_gate = self.__gates.iloc[0].copy(), self.__gates.iloc[-1].copy()

        new_initial_gate.iloc[idx] -= direction_vector * distance_max_meter
        new_final_gate.iloc[idx] += direction_vector * distance_max_meter

        self.__gates = pd.concat([pd.DataFrame([new_initial_gate], columns=self.__gates.columns, index=['fakeInit']), self.__gates, pd.DataFrame([new_final_gate], columns=self.__gates.columns, index=['fakeFinal'])])

        
    def prepareTrajectories(self, numPoints:int=-1, gates:pd.DataFrame=None) -> pd.DataFrame:
        """Prepares the trajectory data for interpolation by normalizing the gate coordinates and applying a spline interpolation to generate a smooth trajectory.
        The method performs the following steps:
        1. Normalizes the gate coordinates by removing any non-numeric characters and scaling the data to have zero mean and unit variance.
        2. Applies a spline interpolation to the normalized gate coordinates to generate a smooth trajectory with 200 points.
        3. Denormalizes the interpolated trajectory by scaling it back to the original mean and standard deviation of the gate coordinates.
        
        Args:
            numPoints (int, optional): The number of points to generate in the interpolated trajectory. If 0, the number of points will be determined based on the number of gates. Defaults to 200.
            gates (pd.DataFrame, optional): A DataFrame containing the gate coordinates. If None, the internal gates DataFrame is used. Defaults to None.

        Returns:
            path (pd.DataFrame): A DataFrame containing the interpolated trajectory points, where each row corresponds to a point in the format [WGS84_Lat2, WGS84_Lon2, Quota Orto. [m]].
        """
        cols = gates.columns if gates is not None else self.__gates.columns
        gates = gates.values.copy() if gates is not None else self.__gates.values.copy()
        
        # 1 - Normalization of the data 
        if np.any(self.__gates.dtypes == 'str'):
            raise ValueError("The gates DataFrame contains non-numeric values. Please ensure that all gate coordinates are numeric before calling prepareTrajectories.")    
            
        gates = gates.astype(float)
        
        mean, std = np.mean(gates, axis=0), np.std(gates, axis=0)
        gates = (gates - mean) / std

        tck, _ = make_splprep(gates.T.tolist(), s=0)

        return pd.DataFrame(np.array([*tck(np.linspace(0, 1, numPoints if numPoints > 0 else (len(self.__gates) * 15)))]).T @ np.diag(std) + mean, columns=cols)
        
    # %% - NEW TRAJECTORIES GENERATION -
    def generateNewTrajectories(self, numTrajectories:int=10, maxDistanceMeters:float=8, startLeft:bool=True) -> list[pd.DataFrame]:
        """Generates a series of new trajectories by adding noise to the original gates positions and then interpolating.
        1. For each trajectory to be generated, random noise is added to the original gate positions. The noise is generated from an exponential distribution with rate parameter (lambdaExp).
        2. The noisy gate positions are then used to create a new trajectory by calling the prepareTrajectories method, which performs normalization and spline interpolation.
        
        Args:
            numTrajectories (int, optional): The number of new trajectories to generate. Defaults to 10.
            maxDistanceMeters (float, optional): The maximum distance in meters for the generated noise. Defaults to 8 meters.
            startLeft (bool, optional): A boolean flag indicating whether to add noise in a leftward direction (True) or rightward direction (False) relative to the original trajectory. Defaults to True.

        Returns:
            newTrajectories (list[pd.DataFrame]): A list of DataFrames, where each DataFrame contains the points of a generated trajectory in the format [WGS84_Lat2, WGS84_Lon2, Quota Orto. [m]].
        """
        trajectories = []
        # Generate noise for each gate and trajectory, more specifically, 
        # if startLeft is True, then the first gate will be pefturbated to the left, the second to the right, and so on, alternating the direction of the noise for each gate.
        # the signs is such that if startLeft is True, then we start with a negative noise for the first gate (pushing it to the left), then a positive noise for the second gate (pushing it to the right), 
        # and so on, alternating the direction of the noise for each gate.
        # Examples: 
        # startLeft = True -> signs = [-1, 1, -1, 1, ...] (first gate left, second gate right, etc.)
        # startLeft = False -> signs = [1, -1, 1, -1, ...] (first gate right, second gate left, etc.)
        signs = np.zeros((self.__gates.values.shape[0], 2))  # Initialize signs with zeros
        signs[1:-1] = (-1) ** (np.arange(len(self.__gates)-2) + int(startLeft)).reshape(-1, 1).repeat(2, axis=1)  # Alternating signs for left and right noise, excluding fake gates

        # 1 degree of latitude is roughly 111.320 km and 1 degree of longitude is roughly 111.320*cos(latitude) km, so we can convert the noise from meters to degrees by dividing by these factors.
        
        # What do we do here:
        # We create the noise 
        for _ in range(numTrajectories):    
            noise = np.random.pareto(a=3, size=(signs.shape[0], 2)) 
            noise[1:-1, :] = (noise[1:-1, :] - noise[1:-1, :].min(axis=0)) / noise[1:-1, :].max(axis=0)   # Apply alternating signs to the noise, excluding fake gates
            noise[1:-1, :] *= signs[1:-1, :] * maxDistanceMeters / np.array([111320, 111320 * np.cos(np.radians(self.__gates['WGS84_Lat2'].mean()))])  # Scale the noise to the desired maximum distance in meters, converting from meters to degrees, excluding fake gates
            gates = self.__gates.copy()
            gates[['WGS84_Lat2', 'WGS84_Lon2']] += noise  # Add noise to the gate positions
            trajectories.append(gates[['WGS84_Lat2', 'WGS84_Lon2', 'Quota Orto. [m]']])  # Prepare the trajectory with the new noisy gates and add it to the list of trajectories

        return trajectories
    

    def saveNewTrajectories(self, trajectories:list[pd.DataFrame], saveName:str) -> None:
        """Saves the generated trajectories to a csv file.
        
        Args:
            trajectories (list[pd.DataFrame]): A list of DataFrames, where each DataFrame contains the points of a generated trajectory in the format [WGS84_Lat2, WGS84_Lon2, Quota Orto. [m]].
            saveName (str): The name of the CSV file to save the trajectories. If the name does not end with '.csv', the method will automatically append the '.csv' extension to the filename.
        """
        if "." in saveName:
            assert saveName.endswith(".csv"), "The filename must end with .csv extension."
            saveName = saveName.split(".csv")[0]

        for i, trajectory in enumerate(trajectories, start=1):
            trajectory.to_csv(os.path.join(self.__picturesSavePath, f"{saveName}_trajectory_{i}.csv"), index=False)



    # %% - VISUALIZATION -
    def plotTrajectory(self, filename:str) -> None:
        """Visualizes the trajectory by creating a map with markers for each gate, including the fake initial and final gates, and saves the map as an HTML file.
        The method uses the Folium library to create an interactive map centered around the average latitude and longitude of the gates, with a zoom level of 30. Each gate is represented as a marker
        with a popup displaying the orthometric height (Quota Orto. [m]). The fake initial and final gates are highlighted with yellow markers, while the other gates are marked with default markers."""
        if "." in filename:
            assert filename.endswith(".html"), "The filename must end with .html extension."
        else:
            filename = filename + ".html"

        folium_map = folium.Map(location=[self.__gates['WGS84_Lat2'].mean(), self.__gates['WGS84_Lon2'].mean()], zoom_start=20)
        for _, row in self.__gates.iterrows():
            if row.name in ['fakeInit', 'fakeFinal']:
                folium.Marker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Orto. [m]']} m", 
                        icon=folium.Icon(color='purple')).add_to(folium_map)
            else:
                folium.Marker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Orto. [m]']} m").add_to(folium_map)

        folium_map.save(os.path.join(self.__picturesSavePath, filename))


    def plotAll(self, data:str, saveName:str) -> None:
        """Visualizes the trajectory along with the original GNSS data by creating a map with markers for each gate and GNSS point, and saves the map as an HTML file.
        The method uses the Folium library to create an interactive map centered around the average latitude and longitude of the gates, with a zoom level of 20. Each gate is represented as 
        a marker with a popup displaying the orthometric height (Quota Orto. [m]). The fake initial and final gates are highlighted with purple markers, while the other gates are marked with red markers. 
        The original GNSS data points are marked with green markers.
        
        Args:
            data (str): The file path to the JSON file containing the original GNSS data, which should have a structure where 'gnss_data' contains 'lat' and 'lon' lists representing the latitude and longitude of the GNSS points, respectively.
            saveName (str): The name of the HTML file to save the plot. If the name does not end with '.html', the method will automatically append the '.html' extension to the filename.
        """
        if "." in saveName:
            assert saveName.endswith(".html"), "The filename must end with .html extension."
        else:
            saveName = saveName + ".html"

        folium_map = folium.Map(location=[self.__gates['WGS84_Lat2'].mean(), self.__gates['WGS84_Lon2'].mean()], zoom_start=20)
        for _, row in self.__gates.iterrows():
            if row.name in ['fakeInit', 'fakeFinal']:
                folium.CircleMarker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Orto. [m]']} m", 
                        radius=2, color="purple", fill=True, fill_color="purple").add_to(folium_map)
            else: 
                folium.CircleMarker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Orto. [m]']} m", 
                        radius=2, color="red", fill=True, fill_color="red").add_to(folium_map)
            
        for _, point in self.prepareTrajectories().iterrows():
            folium.CircleMarker(location=[point['WGS84_Lat2'], point['WGS84_Lon2']], popup=f"Quota: {point['Quota Orto. [m]']} m", 
                        radius=1, color="blue", fill=True, fill_color="blue").add_to(folium_map)

        folium_map.save(os.path.join(self.__picturesSavePath, saveName.split(".html")[0] + "_with_gates.html"))

        with open(data, "r", encoding="UTF-8") as fp:
            data = json.load(fp)

        for lat, lon in zip(data['gnss_data']['lat'], data['gnss_data']['lon']):
            folium.CircleMarker(location=[lat, lon], radius=1, color='green', fill=True, fill_opacity=0.6, fill_color="green").add_to(folium_map)

        folium_map.save(os.path.join(self.__picturesSavePath, saveName.split(".html")[0] + "_with_gnss.html"))


    def plotSimulatedTrajectories(self, trajectories:list[pd.DataFrame], saveName:str) -> None:
        """Visualizes the simulated trajectories by creating a map with markers for each gate and simulated trajectory point, and saves the map as an HTML file.
        The method uses the Folium library to create an interactive map centered around the average latitude and longitude of the gates, with a zoom level of 20. Each gate is represented as 
        a marker with a popup displaying the orthometric height (Quota Orto. [m]). The fake initial and final gates are highlighted with purple markers, while the other gates are marked with red markers. 
        The simulated trajectory points are marked with blue markers.
        
        Args:
            trajectories (list[pd.DataFrame]): A list of DataFrames, where each DataFrame contains the points of a simulated trajectory in the format [WGS84_Lat2, WGS84_Lon2, Quota Orto. [m]].
            saveName (str): The name of the file to save the plot as an HTML file. If the name does not end with '.html', the method will automatically append the '.html' extension to the filename.
        """
        if "." in saveName:
            assert saveName.endswith(".html"), "The filename must end with .html extension."
            saveName = saveName.split(".html")[0]


        for i, trajectory in enumerate(trajectories, start=1):    
            folium_map = folium.Map(location=[self.__gates['WGS84_Lat2'].mean(), self.__gates['WGS84_Lon2'].mean()], zoom_start=20)
            for _, row in self.__gates.iterrows():
                if row.name in ['fakeInit', 'fakeFinal']:
                    folium.CircleMarker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Orto. [m]']} m", 
                            radius=2, color="purple", fill=True, fill_color="purple").add_to(folium_map)
                else: 
                    folium.CircleMarker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Orto. [m]']} m", 
                            radius=2, color="red", fill=True, fill_color="red").add_to(folium_map)

            for _, point in trajectory.iterrows():
                folium.CircleMarker(location=[point['WGS84_Lat2'], point['WGS84_Lon2']], popup=f"Quota: {point['Quota Orto. [m]']} m", 
                        radius=1, color="blue", fill=True, fill_color="blue").add_to(folium_map)

            folium_map.save(os.path.join(self.__picturesSavePath, f"{saveName}_trajectory_{i}.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Loader and Visualizer")
    parser.add_argument("--path", type=str, default="./data/pointsLocationFirstCourse.csv", help="Path to the CSV file containing gate positions.")
    parser.add_argument("--plotGatesPath", type=str, default=None, help="Path to save the plot of gates (HTML file). If not provided, the plot will not be saved.")
    parser.add_argument("--plotInterpolatedTrajectoryPath", type=str, default=None, help="Path to save the plot of the interpolated trajectory (HTML file). If not provided, the plot will not be saved.")
    parser.add_argument("--numTrajectories", type=int, default=-1, help="Number of new trajectories to generate.")
    parser.add_argument("--maxDistanceMeters", type=float, default=8, help="Maximum distance in meters for the generated noise.")
    parser.add_argument("--startLeft", action='store_true', help="Flag to indicate whether to start adding noise to the left of the original trajectory.")
    parser.add_argument("--simulatedTrajectoriesPath", type=str, default=None, help="Path to save the simulated trajectories (CSV files). If not provided, the trajectories will not be saved.")
    parser.add_argument("--plotSimulatedTrajectoriesPath", type=str, default=None, help="Path to save the plot of simulated trajectories (HTML files). If not provided, the plots will not be saved.")
    args = parser.parse_args()

    t = trajectoryLoader(args.path)
    if args.plotGatesPath:
        t.plotTrajectory(args.plotGatesPath)

    if args.plotInterpolatedTrajectoryPath:
        t.plotAll(args.plotInterpolatedTrajectoryPath, args.plotInterpolatedTrajectoryPath.split(".html")[0] + "_with_gates.html")

    if args.simulatedTrajectoriesPath and args.numTrajectories > 0:
        newTraj = t.generateNewTrajectories(numTrajectories=args.numTrajectories, maxDistanceMeters=args.maxDistanceMeters, startLeft=args.startLeft)
        t.plotSimulatedTrajectories(newTraj, args.simulatedTrajectoriesPath)
    



# if __name__ == "__main__":
#     path = "./data/pointsLocationFirstCourse.csv"
#     t = trajectoryLoader(path)
#     t.plotTrajectory()
#     t.plotAll("./data/processed/F_tr1_d1.json")

#     newTraj = t.generateNewTrajectories(numTrajectories=5, maxDistanceMeters=8, startLeft=True)
#     t.plotSimulatedTrajectories(newTraj)

