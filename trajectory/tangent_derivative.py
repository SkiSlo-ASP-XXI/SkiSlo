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

import numpy as np
from scipy.spatial import cKDTree

import laspy


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


def tangent_derivatives(xy, tangents, surface_xyz, search_radius, min_neighbours):
    """Directional derivative of the surface along the trajectory tangent.

    Returns
    -------
    deriv  : (N,)  array   D(s) = grad(z) . t_hat   (NaN where under-sampled).
    grads  : (N, 2) array  the fitted (dz/dx, dz/dy) per point (NaN where skipped).
    """
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
        deriv[i] = a * tangents[i, 0] + b * tangents[i, 1]

    return deriv, grads


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
        default="trajectory/output.las",
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
        xy, tangents, surface, args.search_radius, args.min_neighbours
    )

    valid = np.isfinite(deriv)
    print(
        f"  {valid.sum()}/{len(deriv)} points evaluated; "
        f"D range [{np.nanmin(deriv):.4f}, {np.nanmax(deriv):.4f}], "
        f"mean {np.nanmean(deriv):.4f}"
    )

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