import laspy
import numpy as np
import open3d as o3d
import pyvista as pv
import json


def get_steepness_at_points(query_points, pcd, output_file, search_radius=5.0):
    # 1. Setup KDTree for fast neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    results = []

    for pt in query_points:
        # Shift point to match your centered point cloud
        pt_shifted = pt - [centroid[0], centroid[1], 0]
        
        # 2. Find neighbors around the query point
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pt_shifted, search_radius)
        
        if k < 3:
            results.append(None)
            continue

        # 3. Estimate local normal for these specific neighbors
        # (Or extract existing normals if already computed)
        points_subset = np.asarray(pcd.points)[idx]
        subset_pcd = o3d.geometry.PointCloud()
        subset_pcd.points = o3d.utility.Vector3dVector(points_subset)
        subset_pcd.estimate_normals()
        
        # Average normal of the local patch
        avg_normal = np.mean(np.asarray(subset_pcd.normals), axis=0)
        avg_normal /= np.linalg.norm(avg_normal) # Ensure unit length

        # 4. Calculate Steepness (Angle from vertical)
        # Using degrees for readability
        steepness = np.degrees(np.arccos(np.abs(avg_normal[2]))) 

        # 5. Calculate Direction (Aspect)
        # 0 is East, 90 is North, 180 is West, 270 is South
        direction = np.degrees(np.atan2(avg_normal[1], avg_normal[0]))

        results.append({"point": pt,"steepness": steepness, "aspect": direction})


    # 6. Save results to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

LASFILE = "output.las"
TRAJECTORY_FILE = "trajectory.json"

#using pyvista
las = laspy.read(LASFILE)
points = np.vstack((las.x,las.y,las.z)).T
cloud = pv.PolyData(points)
cloud['Elevation'] = points[:,2]
# Funzione per normalizzare colori 16-bit in [0,1]
def to_float(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.max() > 1.0:  # tipicamente 0..65535
        a = a / 65535.0
    return a


# Se RGB disponibile
if all(hasattr(las, ch) for ch in ("red", "green", "blue")):
    colors = np.column_stack([
        to_float(las.red),
        to_float(las.green),
        to_float(las.blue)
    ])
else:
    # fallback: usa l’elevazione come scala di grigi
    elev = points[:, 2]
    norm = (elev - elev.min()) / (elev.ptp() if elev.ptp() else 1)
    colors = np.column_stack([norm, norm, norm])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
# Compute centroid (mean position)
centroid = points.mean(axis=0)

# Subtract only X and Y components
points[:, 0] -= centroid[0]  # shift X
points[:, 1] -= centroid[1]  # shift Y
# leave Z as is

# Update cloud
pcd.points = o3d.utility.Vector3dVector(points)

print("New centroid (X,Y):", np.asarray(pcd.points).mean(axis=0)[:2])

print("Points imported")
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=1)

#o3d.visualization.draw_geometries([downpcd])

print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
#o3d.visualization.draw_geometries([downpcd], point_show_normal=True)
min_bound = pcd.get_min_bound()  # [x_min, y_min, z_min]
max_bound = pcd.get_max_bound()  # [x_max, y_max, z_max]

print("Min bound:", min_bound)
print("Max bound:", max_bound)


