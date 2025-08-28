import laspy
import numpy as np
import open3d as o3d

las_path = "segment.las"
voxel_size = 2
x_gap = 2.0

def rgb_float01(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.max() > 1.0:
        a /= 65535.0
    return a

las = laspy.read(las_path)
points = np.vstack((las.x, las.y, las.z)).T

if all(hasattr(las, c) for c in ("red", "green", "blue")):
    colors = np.column_stack([
        rgb_float01(las.red),
        rgb_float01(las.green),
        rgb_float01(las.blue),
    ])
else:
    z = points[:, 2]
    nz = (z - z.min()) / (z.ptp() if z.ptp() else 1)
    colors = np.stack([nz, nz, nz], axis=1).astype(np.float32)

pcd_full = o3d.geometry.PointCloud()
pcd_full.points = o3d.utility.Vector3dVector(points)
pcd_full.colors = o3d.utility.Vector3dVector(colors)

pcd_vox = pcd_full.voxel_down_sample(voxel_size=voxel_size)

n_full = len(pcd_full.points)
n_vox  = len(pcd_vox.points)

# Shift voxel cloud in +X to be side-by-side
full_extent_x = pcd_full.get_max_bound()[0] - pcd_full.get_min_bound()[0]
pcd_vox.translate([full_extent_x + x_gap, 0, 0])

# Show in one window; point size via render option
vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f"Full: {n_full:,} | Voxel({voxel_size} m): {n_vox:,}", width=1600, height=900)
vis.add_geometry(pcd_full)
vis.add_geometry(pcd_vox)

opt = vis.get_render_option()
opt.background_color = np.array([1, 1, 1])
opt.point_size = 2.0

vis.run()
vis.destroy_window()


