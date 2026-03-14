import json, os
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev 



class trajectoryLoader():
    def __init__(self, path:str)->None:
        self.__gates = pd.read_csv(path, sep=";", header=0, index_col=0)[['WGS84_Lat2', 'WGS84_Lon2', 'Quota Elliss. WGS84 [m]']] 
        #TODO: verifica quota!!!!

    def prepareTrajectories(self):
        gates = self.__gates.values

        # 1 - Normalization of the data 
        if np.any(self.__gates.dtypes == 'str'):
            gates[:, -1] = np.char.replace(gates[:, -1].astype(str), ".", "").astype(float)
            #gates[:, -1].astype(str).apply(lambda x: float(x.replace(".", ""))).values

        gates = gates.astype(float)

        self.mean, self.std = np.mean(gates, axis=0), np.std(gates, axis=0)
        gates = (gates - self.mean) / self.std

        tck, u = splprep(gates.T.tolist(), s=0)
        u_fine = np.linspace(0, 1, 200)
        x_fine, y_fine, z_fine = splev(u_fine, tck)

        arr = np.array([x_fine, y_fine, z_fine])

        return arr.T @ np.diag(self.std) + self.mean
        



    def plotTrajectory(self):
        folium_map = folium.Map(location=[self.__gates['WGS84_Lat2'].mean(), self.__gates['WGS84_Lon2'].mean()], zoom_start=10)
        for _, row in self.__gates.iterrows():
            folium.Marker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Elliss. WGS84 [m]']} m").add_to(folium_map)

        folium_map.save("trajectory_map.html")


    def plotAll(self):
        folium_map = folium.Map(location=[self.__gates['WGS84_Lat2'].mean(), self.__gates['WGS84_Lon2'].mean()], zoom_start=10)
        for _, row in self.__gates.iterrows():
            folium.CircleMarker(location=[row['WGS84_Lat2'], row['WGS84_Lon2']], popup=f"Quota: {row['Quota Elliss. WGS84 [m]']} m", 
                        radius=2, color="red", fill=True, fill_color="red").add_to(folium_map)
            
        for point in self.prepareTrajectories():
            folium.CircleMarker(location=[point[0], point[1]], popup=f"Quota: {point[2]} m", 
                        radius=1, color="blue", fill=True, fill_color="blue").add_to(folium_map)

        folium_map.save("all_trajectory_map_v1.html")

        data = "./data/processed/F_tr1_d1.json"

        with open(data, "r", encoding="UTF-8") as fp:
            data = json.load(fp)

        for lat, lon in zip(data['gnss_data']['lat'], data['gnss_data']['lon']):
            folium.CircleMarker(location=[lat, lon], radius=1, color='green', fill=True, fill_opacity=0.6, fill_color="green").add_to(folium_map)

        folium_map.save("all_trajectory_map_v2.html")





if __name__ == "__main__":
    path = "./data/pointsLocationFirstCourse.csv"
    pippo = trajectoryLoader(path).prepareTrajectories()
    trajectoryLoader(path).plotTrajectory()
    print(pippo)
    print(pippo.shape)

    folium_map = folium.Map(location=[pippo[:, 0].mean(), pippo[:, 1].mean()], zoom_start=10)
    for point in pippo:
        folium.Marker(location=[point[0], point[1]], popup=f"Quota: {point[2]} m").add_to(folium_map)
    folium_map.save("new_trajectory_map.html")

    trajectoryLoader(path).plotAll()
    
    #TODO: QUOTA!!!
    #TODO: aggiungere punto di partenza e arrivo
    #TODO: aggiungere rumore verso esterno curva per generare nuove traj