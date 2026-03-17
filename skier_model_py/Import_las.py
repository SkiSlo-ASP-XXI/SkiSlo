import laspy
import numpy as np
import open3d as o3d
import pyvista as pv


#using pyvista

las = laspy.read("segment.las")
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

'''
plotter = pv.Plotter()
plotter.add_mesh(cloud,point_size=2,render_points_as_spheres=True,scalars="Elevation",cmap="terrain")
plotter.show()
'''

'''
mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
o3d.visualization.draw(mesh, raw_mode=True)

#using open3d for resampling
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
print(pcd)
o3d.visualization.draw_geometries([pcd])
'''


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

o3d.visualization.draw_geometries([pcd])
print("Points imported")
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=1)
o3d.visualization.draw_geometries([downpcd])

print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
o3d.visualization.draw_geometries([downpcd], point_show_normal=True)
min_bound = pcd.get_min_bound()  # [x_min, y_min, z_min]
max_bound = pcd.get_max_bound()  # [x_max, y_max, z_max]

print("Min bound:", min_bound)
print("Max bound:", max_bound)

