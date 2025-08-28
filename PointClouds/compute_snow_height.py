import numpy as np
import open3d as o3d
import laspy
import matplotlib.pyplot as plt
from matplotlib import cm

GROUND_LAS = "segment.las"
SNOW_PCD   = "segment_snow.pcd"

def rgb_float01(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.max() > 1.0:
        a /= 65535.0
    return a

def load_ground_las(path):
    las = laspy.read(path)
    P = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    if all(hasattr(las, c) for c in ("red","green","blue")):
        C = np.column_stack([
            rgb_float01(las.red),
            rgb_float01(las.green),
            rgb_float01(las.blue),
        ])
    else:
        C = None
    return P, C

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    P = np.asarray(pcd.points, dtype=np.float64)
    return pcd, P

def compute_depth_indexwise(Pg, Ps):
    if Pg.shape[0] != Ps.shape[0]:
        return None
    xy_diff = np.abs(Pg[:, :2] - Ps[:, :2]).max()
    if xy_diff > 1e-6:
        return None
    return (Ps[:, 2] - Pg[:, 2]).astype(np.float32)

def compute_depth_nn_xy(Pg, Ps):
    Pg_xy = Pg.copy(); Pg_xy[:, 2] = 0.0
    Ps_xy = Ps.copy(); Ps_xy[:, 2] = 0.0

    pcd_g_xy = o3d.geometry.PointCloud()
    pcd_g_xy.points = o3d.utility.Vector3dVector(Pg_xy)
    kdt = o3d.geometry.KDTreeFlann(pcd_g_xy)

    depth = np.empty(Ps.shape[0], dtype=np.float32)
    for i, q in enumerate(Ps_xy):
        _, idxs, _ = kdt.search_knn_vector_3d(q, 1)
        j = idxs[0]
        depth[i] = Ps[i, 2] - Pg[j, 2]
    return depth

def apply_colormap(pcd, values, vmin=None, vmax=None, cmap_name="turbo"):
    if vmin is None: vmin = float(np.nanmin(values))
    if vmax is None: vmax = float(np.nanmax(values))
    normed = (values - vmin) / (vmax - vmin + 1e-6)
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(normed)[:, :3]  # RGBA → RGB
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return vmin, vmax, cmap

def save_colorbar(vmin, vmax, cmap, filename="colorbar.png"):
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cbar = plt.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        cax=ax, orientation='horizontal'
    )
    cbar.set_label("Snow depth [m]")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Colorbar saved to {filename}")

def main():
    Pg, _ = load_ground_las(GROUND_LAS)
    pcd_s, Ps = load_pcd(SNOW_PCD)

    depth = compute_depth_indexwise(Pg, Ps)
    if depth is None:
        print("[Info] Falling back to nearest-neighbor in XY.")
        depth = compute_depth_nn_xy(Pg, Ps)

    valid = np.isfinite(depth)
    print(f"Depth stats (m): min={depth[valid].min():.3f}, max={depth[valid].max():.3f}, mean={depth[valid].mean():.3f}")

    vmin, vmax, cmap = apply_colormap(pcd_s, depth, cmap_name="turbo")
    save_colorbar(vmin, vmax, cmap, filename="snow_depth_colorbar.png")

    # Shift snow cloud for comparison
    dx = (np.max(Pg[:,0]) - np.min(Pg[:,0])) * 1.2
    pcd_s_vis = o3d.geometry.PointCloud(pcd_s)
    pcd_s_vis.translate([dx, 0, 0])

    # Show only snow (colored by depth) or both ground+snow side by side
    o3d.visualization.draw_geometries(
        [pcd_s_vis],
        window_name="Snow colored by depth",
        width=1600, height=900
    )

if __name__ == "__main__":
    main()

