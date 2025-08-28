import laspy
import numpy as np
import open3d as o3d

# ---------------- SETTINGS ----------------
las_path      = "segment.las"
voxel_coarse  = 2     # voxel globale (m)
voxel_fine    = 0.8     # voxel locale nelle ROI (m)
# Centri ROI e raggio in coordinate RICENTRATE (dopo centering XY):
rois_xyr = [
    (0.0, 0.0, 10),     # es: 2 m attorno all'origine
    (40, 40, 5),    # altro esempio
]
# ------------------------------------------

def rgb_float01(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.max() > 1.0:  # tipico LAS 16-bit
        a /= 65535.0
    return a

def recenter_xy(points):
    """Ritorna: points_shifted (XY recentrato), shift_xy (centro sottratto)."""
    xy_centroid = points[:, :2].mean(axis=0)  # [cx, cy]
    shifted = points.copy()
    shifted[:, 0] -= xy_centroid[0]
    shifted[:, 1] -= xy_centroid[1]
    return shifted, xy_centroid  # shift_xy = [cx, cy]

def pcd_from_np(P, C):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float32))
    return pcd

def fine_voxel_in_roi(P_centered, C, center_xy, radius, v_fine):
    """Seleziona punti in ROI su XY (nel sistema ricentrato) e li voxelizza fine."""
    cx, cy = center_xy
    r2 = radius * radius
    mask = (P_centered[:,0] - cx)**2 + (P_centered[:,1] - cy)**2 <= r2
    if not np.any(mask):
        return None
    p_roi = P_centered[mask]
    c_roi = C[mask]
    pcd_roi = pcd_from_np(p_roi, c_roi)
    return pcd_roi.voxel_down_sample(voxel_size=v_fine)

def remove_points_in_roi(pcd, center_xy, radius):
    """Rimuove dal pcd i punti dentro la ROI (su XY, sistema ricentrato)."""
    P = np.asarray(pcd.points)
    cx, cy = center_xy
    r2 = radius * radius
    keep = (P[:,0] - cx)**2 + (P[:,1] - cy)**2 > r2
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(P[keep])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])
    return out

# ---------- 1) Leggi LAS ----------
las = laspy.read(las_path)
P_full = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

if all(hasattr(las, c) for c in ("red","green","blue")):
    C_full = np.column_stack([
        rgb_float01(las.red),
        rgb_float01(las.green),
        rgb_float01(las.blue),
    ])
else:
    z = P_full[:,2]
    g = (z - z.min()) / (z.ptp() if z.ptp() else 1)
    C_full = np.stack([g, g, g], axis=1).astype(np.float32)

# ---------- 2) Ricentra solo XY ----------
P_centered, shift_xy = recenter_xy(P_full)  # sottratto [cx, cy]
print(f"Applied XY shift (subtracted): {shift_xy}  --> now origin is at data XY-centroid")

# Nota: ROIs sono definite in questo sistema ricentrato, quindi NON serve trasformarle.

# ---------- 3) Voxel globale ----------
pcd_full_centered = pcd_from_np(P_centered, C_full)
pcd_coarse = pcd_full_centered.voxel_down_sample(voxel_size=voxel_coarse)

# ---------- 4) Raffina localmente ----------
pcd_refined = pcd_coarse
for (cx, cy, rad) in rois_xyr:
    pcd_refined = remove_points_in_roi(pcd_refined, (cx, cy), rad)
    pcd_roi_fine = fine_voxel_in_roi(P_centered, C_full, (cx, cy), rad, voxel_fine)
    if pcd_roi_fine is not None:
        pcd_refined += pcd_roi_fine

# Dedup leggero sui bordi tra coarse/fine
pcd_refined = pcd_refined.voxel_down_sample(voxel_size=min(voxel_fine, voxel_coarse)*0.2)

# ---------- 5) Report ----------
n_full    = len(P_full)
n_coarse  = np.asarray(pcd_coarse.points).shape[0]
n_refined = np.asarray(pcd_refined.points).shape[0]
print(f"Full:     {n_full:,} pts")
print(f"Coarse:   {n_coarse:,} pts (voxel={voxel_coarse} m)")
print(f"Refined:  {n_refined:,} pts (fine={voxel_fine} m in {len(rois_xyr)} ROIs)")

# ---------- 6) Visualizza side-by-side in Open3D ----------
extent_x = pcd_coarse.get_max_bound()[0] - pcd_coarse.get_min_bound()[0]
gap = 0.5 * extent_x if extent_x > 0 else 5.0
pcd_refined_shifted = o3d.geometry.PointCloud(pcd_refined)  # shallow copy
pcd_refined_shifted.translate([extent_x + gap, 0, 0])

geoms = [pcd_coarse, pcd_refined_shifted]

# (opzionale) disegna dischi sottili a terra per mostrare le ROI
try:
    z0 = float(np.min(P_centered[:,2]))
    for (cx, cy, rad) in rois_xyr:
        for xshift in (0.0, extent_x + gap):
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=rad, height=0.001)
            cyl.compute_vertex_normals()
            cyl.paint_uniform_color([1.0, 0.2, 0.2])
            T = np.eye(4)
            T[:3,3] = [cx + xshift, cy, z0]
            cyl.transform(T)
            geoms.append(cyl)
except Exception:
    pass

o3d.visualization.draw_geometries(
    geoms,
    window_name=f"Centered XY | Coarse {voxel_coarse} m vs Refined {voxel_fine} m",
    width=1600, height=900
)

# --------- Nota sulle trasformazioni ----------
# - Tutte le selezioni ROI sono in coordinate RICENTRATE (dopo sottrazione shift_xy).
# - Se vuoi tornare alle coordinate originali di un punto p_centered: p_orig = p_centered + [shift_xy[0], shift_xy[1], 0]
