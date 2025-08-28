import time
import numpy as np
import open3d as o3d
import laspy

# ===================== SETTINGS =====================
las_path      = "segment.las"

# Voxel globale (coarse) e fine
voxel_coarse  = 3        # m
voxel_fine    = 0.5        # m

# Animazione: ROI, tempo, traiettoria (in sistema RICENTRATO XY)
roi_radius    = 5         # m, raggio ROI densa
dt_seconds    = 0.05         # s tra i frame
n_steps       = 100          # numero di step

start_xy      = np.array([-3.0, -2.0])   # punto di partenza (ricentrato)
direction_xy  = np.array([ 1.0,  0.5])   # direzione XY
step_length   = 0.3                      # m per step
# =====================================================

def rgb_float01(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.max() > 1.0:
        a /= 65535.0
    return a

def recenter_xy(points):
    xy_centroid = points[:, :2].mean(axis=0)
    shifted = points.copy()
    shifted[:, 0] -= xy_centroid[0]
    shifted[:, 1] -= xy_centroid[1]
    return shifted, xy_centroid  # centro sottratto

def pcd_from_np(P, C):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float32))
    return pcd

def fine_voxel_in_roi(P_centered, C, center_xy, radius, v_fine):
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
    P = np.asarray(pcd.points)
    cx, cy = center_xy
    r2 = radius * radius
    keep = (P[:,0] - cx)**2 + (P[:,1] - cy)**2 > r2
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(P[keep])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])
    return out

# ---------------- Load LAS & build base clouds ----------------
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

# Recenter XY
P_centered, shift_xy = recenter_xy(P_full)
print(f"[Centering] Applied XY shift (subtracted): {shift_xy}")

pcd_full_centered = pcd_from_np(P_centered, C_full)
pcd_coarse = pcd_full_centered.voxel_down_sample(voxel_size=voxel_coarse)

# ---------------- Path setup ----------------
# Normalizza la direzione e prepara i centri step-by-step
dir_xy = direction_xy.astype(float)
norm = np.linalg.norm(dir_xy)
if norm == 0:
    dir_xy[:] = [1.0, 0.0]
else:
    dir_xy /= norm

centers_xy = [start_xy + i * step_length * dir_xy for i in range(n_steps)]

# ---------------- Animated geometry objects ----------------
# pcd_anim: geometria che aggiorneremo a ogni frame (coarse + fine ROI corrente)
# marker: piccolo sferoide per mostrare il centro ROI corrente
pcd_anim = o3d.geometry.PointCloud()
pcd_anim.points = o3d.utility.Vector3dVector(np.asarray(pcd_coarse.points).copy())
pcd_anim.colors = o3d.utility.Vector3dVector(np.asarray(pcd_coarse.colors).copy())

marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
marker.compute_vertex_normals()
marker.paint_uniform_color([1.0, 0.2, 0.2])

# Funzione per costruire la nuvola "refined" per un dato centro
def build_refined_for_center(center_xy):
    # 1) togli i punti coarse dentro ROI
    refined = remove_points_in_roi(pcd_coarse, center_xy, roi_radius)
    # 2) aggiungi i punti fine dai dati originali (ricentrati)
    pcd_roi_fine = fine_voxel_in_roi(P_centered, C_full, center_xy, roi_radius, voxel_fine)
    if pcd_roi_fine is not None:
        refined += pcd_roi_fine
    # 3) dedup leggero per unire i bordi
    refined = refined.voxel_down_sample(voxel_size=min(voxel_fine, voxel_coarse)*0.2)
    return refined

# Prepara primo stato
ref0 = build_refined_for_center(centers_xy[0])
pcd_anim.points = ref0.points
pcd_anim.colors = ref0.colors

# Posiziona il marker sul centro attuale (Z = quota min della nuvola)
z0 = float(np.min(P_centered[:,2])) if len(P_centered) else 0.0
T = np.eye(4); T[:3,3] = [centers_xy[0][0], centers_xy[0][1], z0]
marker.transform(T)

# ---------------- Animation callback (0.2 s per step) ----------------
state = {
    "i": 0,
    "last_t": time.time()
}

def animation_callback(vis):
    now = time.time()
    if now - state["last_t"] < dt_seconds:
        return False  # aspetta fino a dt

    state["last_t"] = now
    state["i"] += 1
    if state["i"] >= len(centers_xy):
        return False  # fine animazione

    cxy = centers_xy[state["i"]]

    # Ricostruisci refined per il nuovo centro
    refined = build_refined_for_center(cxy)

    # Aggiorna la geometria animata
    pcd_anim.points = refined.points
    pcd_anim.colors = refined.colors
    vis.update_geometry(pcd_anim)

    # Sposta il marker
    # reset transform (ricrea mesh per semplicità)
    vis.remove_geometry(marker, reset_bounding_box=False)
    new_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    new_marker.compute_vertex_normals()
    new_marker.paint_uniform_color([1.0, 0.2, 0.2])
    T = np.eye(4); T[:3,3] = [cxy[0], cxy[1], z0]
    new_marker.transform(T)
    # aggiorna reference globale
    globals()['marker'] = new_marker
    vis.add_geometry(new_marker, reset_bounding_box=False)

    return True  # continua

# ---------------- Launch viewer with animation ----------------
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name=f"Moving ROI densification — coarse={voxel_coarse} m, fine={voxel_fine} m",
                  width=1600, height=900)
vis.add_geometry(pcd_anim)
vis.add_geometry(marker)

opt = vis.get_render_option()
opt.background_color = np.array([1,1,1])
opt.point_size = 2.0

# Inquadra tutto all’inizio
aabb = pcd_coarse.get_axis_aligned_bounding_box()
vis.get_view_control().set_lookat(aabb.get_center())
vis.get_view_control().set_front([0.0, -1.0, 0.3])
vis.get_view_control().set_up([0.0, 0.0, 1.0])
vis.get_view_control().set_zoom(0.7)

# Registra animazione
vis.register_animation_callback(animation_callback)
vis.run()
vis.destroy_window()
