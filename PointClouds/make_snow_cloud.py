# make_snow_cloud.py
import numpy as np
import open3d as o3d
import laspy
from pathlib import Path

# ---------- inputs/outputs ----------
LAS_IN   = "segment.las"          # your bare ground LAS
PCD_OUT  = "segment_snow.pcd"     # simulated snow point cloud
NPY_OUT  = "segment_snow_depth.npy"  # per-point snow depth array (for reference)
# -----------------------------------

def rgb_float01(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.max() > 1.0:  # 16-bit LAS
        a /= 65535.0
    return a

def pcd_from_np(P, C):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    if C is not None:
        p.colors = o3d.utility.Vector3dVector(C.astype(np.float32))
    return p

def main():
    # Load base ground LAS
    las = laspy.read(LAS_IN)
    P = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

    # Colors if present
    if all(hasattr(las, c) for c in ("red","green","blue")):
        C = np.column_stack([rgb_float01(las.red), rgb_float01(las.green), rgb_float01(las.blue)])
    else:
        C = None

    # Simulate snow depth in [0, 1] m per point
    depth = np.random.uniform(0.0, 1.0, size=P.shape[0]).astype(np.float32)

    # Build snow cloud by adding depth to Z
    P_snow = P.copy()
    P_snow[:, 2] += depth

    # Create Open3D point cloud and save
    pcd_snow = pcd_from_np(P_snow, C)
    o3d.io.write_point_cloud(PCD_OUT, pcd_snow, write_ascii=False, compressed=True)
    np.save(NPY_OUT, depth)

    print(f"[OK] Snow cloud saved: {PCD_OUT}  (points: {len(P_snow)})")
    print(f"[OK] Depth array saved: {NPY_OUT}")

if __name__ == "__main__":
    main()
