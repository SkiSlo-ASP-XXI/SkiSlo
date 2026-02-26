import argparse, os
import folium
import matplotlib.pyplot as plt
import numpy as np

from json import load

def plot_map(data:dict, saveMap:str, filename:str):
    folium_map = folium.Map(location=[data['gnss_data']['lat'][0], data['gnss_data']['lon'][0]], zoom_start=15)

    for lat, lon in zip(data['gnss_data']['lat'], data['gnss_data']['lon']):
        folium.CircleMarker(location=[lat, lon], radius=2, color='blue').add_to(folium_map)

    legend_html = f""" <div style=" position: fixed; bottom: 20px; left: 20px; width: 180px; background-color: white;
        border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; "> 
            <b>Legend</b><br> 
            <ul>
            <li>skier = {data['metadata']['skier']}</li>
            <li>track = {data['metadata']['track']}</li>
            <li>run = {data['metadata']['run']}</li>
            <li>date = {data['metadata']['day']} / {data['metadata']['month']} / {data['metadata']['year']}</li>
            </ul>
            </div> """ 

    folium_map.get_root().html.add_child(folium.Element(legend_html))


    folium_map.save(saveMap + f"/{filename}.html")


def plot_speed(data:dict) -> None:
    plt.figure(figsize=(12, 6))
    
    plt.plot(data['gnss_data']['timestamp'], np.array(data['gnss_data']['ground_speed'])*3.6, label='Ground Speed (km/h)', color='blue', linewidth=2)
    avg = np.mean(data['gnss_data']['ground_speed'])*3.6
    plt.axhline(y=avg, color='red', linestyle='--', label=f'Average Speed {avg:.2f} km/h', linewidth=1)
    
    plt.title(f'Ground Speed Over Time for {data["metadata"]["skier"]} on track {data["metadata"]["track"]} descent {data["metadata"]["run"]}', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Speed (km/h)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize skier data from a JSON file.')
    parser.add_argument('--file', required=True, type=str, help='Path to input JSON file (dirs + filename.json).')
    parser.add_argument('--saveMap', type=str, default=None, help='Path to save the map visualization as an HTML file.')
    parser.add_argument('--plotSpeed', action='store_true', help='Plot the skier ground speed over time.')

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise FileNotFoundError(f"File not found: {args.file}")
    
    if args.saveMap:
        if os.path.exists(args.saveMap) and not os.path.isdir(args.saveMap):
            raise NotADirectoryError(f"Path exists but is not a directory: {args.saveMap}")
        os.makedirs(args.saveMap, exist_ok=True)    

    with open(args.file, 'r', encoding='utf-8') as f:
        data = load(f)

    data['metadata']['skier'] = 'Felix' if 'F' in data['metadata']['skier'] else ('Matteo' if 'M' in data['metadata']['skier'] else 'Unknown')

    if args.saveMap:
        plot_map(data, args.saveMap, os.path.basename(args.file).split(".")[0])

    if args.plotSpeed:
        plot_speed(data)
        plt.show()

    