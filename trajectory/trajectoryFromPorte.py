import json, os
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_splprep

class trajectoryLoader():
    VALID_NAMES = ['Est [m]', 'Nord [m]', 'Quota Orto. [m]', 'WGS84_Lat2', 'WGS84_Lon2']

    def __init__(self, path:str, addStarterAndFinisher:bool=True)->None:
        os.makedirs("./proveMapper", exist_ok=True)
        self.__gates = pd.read_csv(path, sep=";", header=0, index_col=0)[self.VALID_NAMES] 
        if addStarterAndFinisher:
            self.__addFakeInitialAndFinalGates()
        

    @property
    def gates(self) -> pd.DataFrame:
        """Returns a copy of the gates DataFrame to prevent external modifications.
        
        Returns:
            portPositions (pd.DataFrame): A copy of the gates DataFrame.
        """
        return self.__gates.copy()


    def __addFakeInitialAndFinalGates(self) -> None:
        """Adds fake initial and final gates to the trajectory based on the existing gates.
        The fake gates are created by extending the trajectory in both directions by a distance equal to the maximum distance between consecutive gates.
        
        This method modifies the internal gates DataFrame by adding two new rows: 'fakeInit' and 'fakeFinal', which represent the extended initial and final gates, respectively.
        """
        idx = self.__gates.columns.get_indexer(['WGS84_Lat2', 'WGS84_Lon2', 'Quota Orto. [m]'])

        distance_max_meter = max(map(lambda x: np.linalg.norm(x[0][idx] - x[1][idx]), zip(self.__gates.iloc[:-1].values, self.__gates.iloc[1:].values)))

        direction_vector = self.__gates.iloc[1, idx] - self.__gates.iloc[0, idx] # Direction from the first gate to the second gate
        direction_vector /= np.linalg.norm(direction_vector)  # Normalize the direction vector

        new_initial_gate = self.__gates.iloc[0].copy()
        new_final_gate = self.__gates.iloc[-1].copy()

        new_initial_gate.iloc[idx] -= direction_vector * distance_max_meter
        new_final_gate.iloc[idx] += direction_vector * distance_max_meter

        self.__gates = pd.concat([pd.DataFrame([new_initial_gate], columns=self.__gates.columns, index=['fakeInit']), 
                                self.__gates, pd.DataFrame([new_final_gate], columns=self.__gates.columns, index=['fakeFinal'])], ignore_index=False)

        
    def prepareTrajectories(self, numPoints:int=-1) -> pd.DataFrame:
        """Prepares the trajectory data for interpolation by normalizing the gate coordinates and applying a spline interpolation to generate a smooth trajectory.
        The method performs the following steps:
        1. Normalizes the gate coordinates by removing any non-numeric characters and scaling the data to have zero mean and unit variance.
        2. Applies a spline interpolation to the normalized gate coordinates to generate a smooth trajectory with 200 points.
        3. Denormalizes the interpolated trajectory by scaling it back to the original mean and standard deviation of the gate coordinates.
        
        Args:
            numPoints (int, optional): The number of points to generate in the interpolated trajectory. If 0, the number of points will be determined based on the number of gates. Defaults to 200.

        Returns:
            path (pd.DataFrame): A DataFrame containing the interpolated trajectory points, where each row corresponds to a point in the format [WGS84_Lat2, WGS84_Lon2, Quota Orto. [m]].
        """
        gates = self.__gates.values.copy()

        # 1 - Normalization of the data 
        if np.any(self.__gates.dtypes == 'str'):
            gates[:, -1] = np.char.replace(gates[:, -1].astype(str), ".", "").astype(float)
            
        gates = gates.astype(float)

        mean, std = np.mean(gates, axis=0), np.std(gates, axis=0)
        gates = (gates - mean) / std

        tck, _ = make_splprep(gates.T.tolist(), s=0)

        return pd.DataFrame(np.array([*tck(np.linspace(0, 1, numPoints if numPoints > 0 else (len(self.__gates) * 15)))]).T @ np.diag(std) + mean, columns=self.__gates.columns)
        

    def plotTrajectory(self) -> None:
        folium_map = folium.Map(location=[self.__gates['WGS84_Lat2'].mean(), self.__gates['WGS84_Lon2'].mean()], zoom_start=30)
        for _, row in self.__gates.iterrows():
            if row.name in ['fakeInit', 'fakeFinal']:
                folium.Marker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Orto. [m]']} m", 
                        icon=folium.Icon(color='purple')).add_to(folium_map)
            else:
                folium.Marker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Orto. [m]']} m").add_to(folium_map)

        folium_map.save("./proveMapper/trajectory_map.html")


    def plotAll(self, data:str) -> None:
        folium_map = folium.Map(location=[self.__gates['WGS84_Lat2'].mean(), self.__gates['WGS84_Lon2'].mean()], zoom_start=30)
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

        folium_map.save("./proveMapper/all_trajectory_map_v1.html")

        with open(data, "r", encoding="UTF-8") as fp:
            data = json.load(fp)

        for lat, lon in zip(data['gnss_data']['lat'], data['gnss_data']['lon']):
            folium.CircleMarker(location=[lat, lon], radius=1, color='green', fill=True, fill_opacity=0.6, fill_color="green").add_to(folium_map)

        folium_map.save("./proveMapper/all_trajectory_map_v2.html")




if __name__ == "__main__":
    path = "./data/pointsLocationFirstCourse.csv"
    trajectoryLoader(path).plotTrajectory()
    trajectoryLoader(path).plotAll("./data/processed/F_tr1_d1.json")
    
    #TODO: aggiungere rumore verso esterno curva per generare nuove traj