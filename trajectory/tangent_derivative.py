"""
Tangent derivative of the surface along a 3D trajectory.

For every point s of a trajectory this computes the directional derivative of the
surface z = f(x, y) in the direction tangent to the surface that follows the
trajectory's horizontal travel direction:

        D(s) = grad(z) . t_hat        [metres of rise per metre travelled]

where
  * grad(z) = (dz/dx, dz/dy) is estimated by fitting a local plane to the
    surface points in a horizontal neighbourhood around the trajectory point, and
  * t_hat is the unit horizontal travel direction of the trajectory at s.

The surface is read from a .las file. Its X/Y are in UTM32N (EPSG:32632), matching
the trajectory's `utm32n_easting` / `utm32n_northing`, so no reprojection is needed.

NOTE on the elevation source
----------------------------
`trajectory/output.las` stores X/Y in UTM32N but its Z channel is NOT a terrain
elevation (it is quantised to [1, 255] and correlates only ~0.63 with the GNSS
height). The pipeline below is geometrically correct, but with that file the
returned values are the slope of *whatever* the Z channel encodes, not the terrain
steepness. To get real terrain slope, point --las at a DTM whose Z is true
elevation; everything else stays the same.

Output
------
The per-point derivative values are collected into a 1-D numpy array (one value per
GNSS point, NaN where the neighbourhood was too small) and saved to disk:
  * <out>.npy            the raw derivative array
  * <out>.csv            easting, northing, height, dz_dx, dz_dy, tangent_x,
                         tangent_y, tangent_derivative  (one row per point)
"""

import argparse
import json
import os

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import (
    binary_fill_holes,
    binary_closing,
    binary_erosion,
    label,
)

import laspy


from scipy.ndimage import convolve

def load_trajectory(json_path):
    """Load the GNSS trajectory as horizontal positions + reference height.

    Returns
    -------
    xy : (N, 2) array   UTM32N easting/northing of each GNSS point.
    z  : (N,)  array    reference orthometric height (height_msl), for output only.
    """
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    g = data["gnss_data"]
    easting = np.asarray(g["utm32n_easting"], dtype=np.float64)
    northing = np.asarray(g["utm32n_northing"], dtype=np.float64)
    height = np.asarray(g["height_msl"], dtype=np.float64)
    return np.column_stack([easting, northing]), height


def unit_tangents(xy):
    """Unit horizontal travel direction at each point via central differences.

    t_hat[i] points along the trajectory from point i-1 to point i+1.
    """
    d = np.gradient(xy, axis=0)               # (N, 2) tangent vectors
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0                   # guard against duplicated points
    return d / norm


def load_surface_points(las_path, bbox, margin):
    """Read the .las surface, keeping only points inside bbox (+margin).

    The full cloud can be hundreds of millions of points; cropping to the
    trajectory's footprint keeps memory and the KDTree small.

    Parameters
    ----------
    bbox : (xmin, ymin, xmax, ymax) of the trajectory.
    margin : metres of padding added around the bbox.

    Returns
    -------
    (M, 3) array of surface points (x, y, z) inside the cropped region.
    """
    xmin, ymin, xmax, ymax = bbox
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin

    kept = []
    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(8_000_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            if mask.any():
                z = np.asarray(chunk.z)
                kept.append(np.column_stack([x[mask], y[mask], z[mask]]))

    if not kept:
        raise RuntimeError(
            "No surface points fall within the trajectory footprint. "
            "Check that the .las and the trajectory share the same CRS."
        )
    return np.concatenate(kept, axis=0)


def obtain_slope_borders(las_path, grid_res=2.0):
    """Recover the outline (borders) of the ski slope from a .las point cloud.

    The .las is assumed to be already cropped to the slope, so the border of the
    3D reconstruction *is* the slope border. The points are rasterised onto a
    horizontal occupancy grid (X/Y in UTM32N); the grid is morphologically cleaned
    (closing + hole filling, keeping the largest connected component) and the
    one-cell-thick outline of the resulting region is returned as UTM32N
    coordinates. Rasterising keeps memory bounded by the grid size regardless of
    how many points the cloud holds.

    Parameters
    ----------
    las_path : str
        Path to the surface .las (X/Y in UTM32N, matching the trajectory CRS).
    grid_res : float
        Size [m] of the occupancy-grid cells. Larger values bridge gaps in a
        sparse cloud at the cost of border precision.

    Returns
    -------
    est, nord : (B,) arrays
        UTM32N easting/northing of the border cells.
    """
    with laspy.open(las_path) as reader:
        hdr = reader.header
        xmin, ymin, xmax, ymax = hdr.x_min, hdr.y_min, hdr.x_max, hdr.y_max

    nx = int(np.ceil((xmax - xmin) / grid_res)) + 1
    ny = int(np.ceil((ymax - ymin) / grid_res)) + 1
    occ = np.zeros((nx, ny), dtype=bool)

    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(8_000_000):
            ix = ((np.asarray(chunk.x) - xmin) / grid_res).astype(np.intp)
            iy = ((np.asarray(chunk.y) - ymin) / grid_res).astype(np.intp)
            np.clip(ix, 0, nx - 1, out=ix)
            np.clip(iy, 0, ny - 1, out=iy)
            occ[ix, iy] = True

    # Consolidate the rasterised cloud into a solid region.
    occ = binary_closing(occ, structure=np.ones((3, 3), dtype=bool))
    occ = binary_fill_holes(occ)

    # Keep only the largest connected component (drop stray specks off the slope).
    lab, n = label(occ)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        occ = lab == sizes.argmax()

    # One-cell-thick outline = region minus its erosion.
    boundary = occ & ~binary_erosion(occ)

    bx, by = np.nonzero(boundary)
    est = xmin + bx * grid_res
    nord = ymin + by * grid_res
    return est, nord


def sample_surface_z(x, y, las_path, search_radius=5.0, min_neighbours=5, margin=10.0):
    """Surface elevation at arbitrary horizontal positions, read from the .las.

    A local plane is fitted to the cloud points within `search_radius` of each
    query position and evaluated there, so the height is interpolated rather than
    snapped to the closest discrete LAS point (as the nearest-neighbour lookup in
    `obtain_inclination` does).

    The .las is cropped to the slope, so a position off the slope has no points
    near it and returns NaN. Callers must treat NaN as "off the reconstructed
    surface": a nearest-neighbour query would instead return the border point's
    height, which is indistinguishable from a real reading.

    Parameters
    ----------
    x, y : array-like    UTM32N easting/northing of the query positions.
    las_path : str       Path to the surface .las (X/Y in UTM32N, Z = height).

    Returns
    -------
    (N,) array of surface heights, NaN where the position is off the slope.
    """
    xy = np.column_stack([np.asarray(x, dtype=np.float64).ravel(),
                          np.asarray(y, dtype=np.float64).ravel()])
    bbox = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    surface = load_surface_points(las_path, bbox, margin + search_radius)
    tree = cKDTree(surface[:, :2])

    z = np.full(len(xy), np.nan)
    for i, q in enumerate(xy):
        idx = tree.query_ball_point(q, r=search_radius)
        if len(idx) < min_neighbours:
            continue
        pts = surface[idx]
        centre = pts[:, :2].mean(axis=0)
        A = np.column_stack([pts[:, 0] - centre[0], pts[:, 1] - centre[1], np.ones(len(pts))])
        try:
            coeffs, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        except np.linalg.LinAlgError:
            continue
        # Plane is fitted in coordinates centred on the patch; evaluate it at q.
        z[i] = coeffs[0] * (q[0] - centre[0]) + coeffs[1] * (q[1] - centre[1]) + coeffs[2]
    return z


def plane_gradient(points):
    """Least-squares fit z = a*x + b*y + c to a local patch; return (a, b).

    (a, b) == (dz/dx, dz/dy) is the surface gradient of the fitted plane.
    Returns (nan, nan) if the patch is degenerate (collinear / too few points).
    """
    xy = points[:, :2]
    z = points[:, 2]
    # Design matrix [x, y, 1]; centre xy for numerical conditioning.
    centre = xy.mean(axis=0)
    A = np.column_stack([xy[:, 0] - centre[0], xy[:, 1] - centre[1], np.ones(len(z))])
    try:
        coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return coeffs[0], coeffs[1]


def tangent_derivatives(xy, surface_xyz, search_radius, min_neighbours, max_desc_dir):
    """Directional derivative of the surface along the trajectory tangent.

    `max_desc_dir` may be given as a 3D vector (e.g. the steepest-descent vector
    of the fitted plane); only its horizontal part is used, and it is normalised
    here. That matters: a 3D-unit descent vector has a horizontal part of norm
    cos(slope), so projecting on it as-is scales D by cos(slope) and
    underestimates the inclination (3.4 deg too low on a 30 deg slope).

    Returns
    -------
    deriv  : (N,)  array   D(s) = grad(z) . t_hat   (NaN where under-sampled).
    grads  : (N, 2) array  the fitted (dz/dx, dz/dy) per point (NaN where skipped).
    """
   # tangents = unit_tangents(xy)

    # D = grad(z) . u is the rise per metre travelled only if u is a horizontal
    # UNIT vector, so tan(alpha) = D and alpha = arctan(D).
    u = np.asarray(max_desc_dir, dtype=np.float64)[:2]
    u = u / (np.linalg.norm(u) + 1e-12)

    tree = cKDTree(surface_xyz[:, :2])        # horizontal KDTree for plane fits
    n = len(xy)
    deriv = np.full(n, np.nan)
    grads = np.full((n, 2), np.nan)

    for i in range(n):
        idx = tree.query_ball_point(xy[i], r=search_radius)
        if len(idx) < min_neighbours:
            continue
        a, b = plane_gradient(surface_xyz[idx])
        if np.isnan(a):
            continue
        grads[i] = (a, b)
        deriv[i] = a * u[0] + b * u[1]

    return deriv, grads

def obtain_inclination(x, y, file_source, desc_vect):
    """Inclination [degrees] of the surface along the trajectory at each point.

    The tangent derivative D = grad(z) . t_hat is the rise per metre travelled,
    i.e. tan(theta), so the inclination angle is theta = arctan(D).

    Returns
    -------
    (N,) array of inclination angles in degrees (NaN where under-sampled).
    """
    xy = np.stack([x, y], axis=1)
    bbox = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    surface = load_surface_points(file_source, bbox, 10.0)
    der, grads = tangent_derivatives(xy, surface, 5.0, 5, desc_vect)

    # Sample the surface height at each trajectory point via nearest neighbour.
    # An exact (x, y) merge never matches: the interpolated trajectory points do
    # not coincide with discrete LAS points, so we look up the closest one.
    tree = cKDTree(surface[:, :2])
    _, nn_idx = tree.query(xy, k=1)
    z_coordinates = surface[nn_idx, 2]

    return np.degrees(np.arctan(der)), grads, z_coordinates

def main():
    parser = argparse.ArgumentParser(
        description="Derivative of the surface along the tangent of a trajectory."
    )
    parser.add_argument(
        "--trajectory",
        default="data/processed/F_tr1_d1.json",
        help="Path to the trajectory JSON (gnss_data with utm32n_easting/northing).",
    )
    parser.add_argument(
        "--las",
        default="/Users/andre/Documents/github.nosync/SkiSlo/data/surfaces/Sestriere_fotogrammetria_95000.las",
        help="Path to the surface .las (X/Y in UTM32N, Z = surface height).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output basename (without extension). "
        "Default: outputs/tangent_derivative_<trajectory-name>.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=5.0,
        help="Horizontal radius [m] of the neighbourhood used for the plane fit.",
    )
    parser.add_argument(
        "--min-neighbours",
        type=int,
        default=3,
        help="Minimum surface points required to fit a plane.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=10.0,
        help="Padding [m] added around the trajectory bbox when cropping the .las.",
    )
    args = parser.parse_args()

    if args.out is None:
        name = os.path.splitext(os.path.basename(args.trajectory))[0]
        args.out = os.path.join("outputs", f"tangent_derivative_{name}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Loading trajectory: {args.trajectory}")
    xy, height = load_trajectory(args.trajectory)
    tangents = unit_tangents(xy)
    print(f"  {len(xy)} GNSS points")

    bbox = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    print(f"Cropping surface from: {args.las}")
    surface = load_surface_points(args.las, bbox, args.margin)
    print(f"  {len(surface)} surface points within footprint")

    print(f"Computing tangent derivatives (radius={args.search_radius} m)")
    deriv, grads = tangent_derivatives(
        xy, surface, args.search_radius, args.min_neighbours
    )

    valid = np.isfinite(deriv)
    print(
        f"  {valid.sum()}/{len(deriv)} points evaluated; "
        f"D range [{np.nanmin(np.degrees(np.arctan(deriv))):.4f}, {np.nanmax(np.degrees(np.arctan(deriv))):.4f}], "
        f"mean {np.nanmean(deriv):.4f}"
    )
    #plot the inclination along the trajectory in degrees
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(np.degrees(np.arctan(deriv)), label="Inclination (degrees)")
    plt.xlabel("Point index")
    plt.ylabel("Inclination (degrees)")
    plt.title("Surface Inclination Along Trajectory")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 1) raw derivative array
    npy_path = args.out + ".npy"
    np.save(npy_path, deriv)
    print(f"Saved derivative array: {npy_path}")

    # 2) detailed CSV
    csv_path = args.out + ".csv"
    table = np.column_stack(
        [xy[:, 0], xy[:, 1], height, grads[:, 0], grads[:, 1],
         tangents[:, 0], tangents[:, 1], deriv]
    )
    header = ("easting,northing,height,dz_dx,dz_dy,"
              "tangent_x,tangent_y,tangent_derivative")
    np.savetxt(csv_path, table, delimiter=",", header=header, comments="")
    print(f"Saved detailed table:  {csv_path}")


if __name__ == "__main__":
    main()