"""3D hazard heatmap from a saved `hazard_coefficients.csv`.

Re-draws the hazard surface of `COEFFICIENTS_EXPERIMENTS.py` with pyvista, without
re-running the trajectory generation and the skier simulation: the per-point
coefficients are read back from the CSV, binned into the same `riverMatrix`, and
draped over the slope rebuilt from the .las.

By default the scene is rendered off-screen and saved as a PNG, so a batch run opens
no window; `--show` opens the interactive pyvista viewer instead (the PNG is still
written, from the camera as it is when the window is closed).

Any coefficient column of the CSV can be plotted (`--column`), so the same file also
gives the simulation / bump / inclination / snow-depth heatmaps on their own;
`--column all` writes one figure per coefficient column.

Usage
-----
    .myvenv/bin/python plot_from_csv.py outputs/simulated_trajectories/coeffs_0_0_0_1/hazard_coefficients.csv
    .myvenv/bin/python plot_from_csv.py <csv> --column "Simulation Coefficient"
    .myvenv/bin/python plot_from_csv.py <csv> --column all
    .myvenv/bin/python plot_from_csv.py <csv> --gates data/pointsLocationFirstCourse.csv --show

Needs pyvista, which is installed in `.myvenv`.

Output
------
    <csv folder>/<column>_surface_3d.png    unless --out is given (single column only)
"""

import argparse
import os

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.ndimage import binary_erosion, convolve

from COEFFICIENTS_EXPERIMENTS import (MAX_TRAJECTORIES_DRAWN, _dem_z_at, build_dem, fill_dem_holes,
                                      get_descent_direction, hazard_on_dem)

DEFAULT_LAS = "data/surfaces/Sestriere_fotogrammetria_95000.las"


def build_river_matrix(df: pd.DataFrame, column: str):
    """Bin the per-point values into the 1 m grid used by COEFFICIENTS_EXPERIMENTS.

    Same construction as the main script (mean per cell, then the 3x3 neighbour fill of
    uncovered cells), so the figure matches the one written during the experiment.

    Returns
    -------
    riverMatrix, reliabilityMatrix, minX, minY, sx, sy
    """
    est, nord = df["Est [m]"].values, df["Nord [m]"].values
    minX, maxX, minY, maxY = est.min(), est.max(), nord.min(), nord.max()

    H, W = int(maxX - minX) + 1, int(maxY - minY) + 1
    xs = ((est - minX) / (maxX - minX + 1e-9) * (W - 1)).astype(np.intp)
    ys = ((nord - minY) / (maxY - minY + 1e-9) * (H - 1)).astype(np.intp)

    flat = ys * W + xs
    river = np.bincount(flat, weights=df[column].values, minlength=H * W).reshape(H, W)
    rel = np.bincount(flat, minlength=H * W).reshape(H, W).astype(float)
    river = np.divide(river, rel, out=np.zeros_like(river), where=rel > 0)

    kernel = np.ones((3, 3))
    rel_sum = convolve(rel, kernel, mode='constant', cval=0)
    mask = (rel == 0) & (convolve(rel.astype(np.int32), kernel.astype(np.int32),
                                  mode='constant', cval=0) >= 4)
    river = np.where(mask, convolve(river * rel, kernel, mode='constant', cval=0)
                     / np.where(rel_sum > 0, rel_sum, 1), river)
    rel = np.where(mask, rel_sum, rel)

    sx = (W - 1) / (maxX - minX + 1e-9)
    sy = (H - 1) / (maxY - minY + 1e-9)
    return river, rel, minX, minY, sx, sy


def load_gates(path):
    """Real gates from the --gates file, or an empty frame when none is given."""
    if path is None:
        return pd.DataFrame({"Est [m]": [], "Nord [m]": []})
    from trajectory.trajectoryFromPorte import trajectoryLoader
    return trajectoryLoader(path).gates.drop(index=['fakeInit', 'fakeFinal'], errors='ignore')


def render_pyvista(x_c, y_c, Z, haz, covered, listDf, gates, max_xy, desc_vect, column,
                   out, mode='extrapolate', z_exag=1.0, show=False):
    """Drape `haz` over the DEM with pyvista and save a screenshot to `out`.

    Coordinates are shifted to the DEM's south-west corner before rendering: VTK works
    in float32, and raw UTM northings (~4.98e6 m) would leave the mesh visibly jittery.
    """
    x0, y0 = float(x_c[0]), float(y_c[0])
    X, Y = np.meshgrid(x_c - x0, y_c - y0)

    # A structured grid needs a vertex everywhere: empty cells are parked at the lowest
    # elevation and hidden, which leaves the real piste outline.
    on_surface = np.isfinite(Z)
    grid = pv.StructuredGrid(X, Y, np.where(on_surface, Z, float(np.nanmin(Z))))
    # StructuredGrid ravels in Fortran order; a C-order scalar would be scrambled.
    grid.hide_points(~on_surface.ravel(order='F'))
    grid[column] = haz.ravel(order='F')

    clim = (float(np.nanmin(haz)), float(np.nanmax(haz))) if np.isfinite(haz).any() else (0.0, 1.0)
    dz_range = float(np.nanmax(Z) - np.nanmin(Z)) if on_surface.any() else 1.0
    lift = 0.01 * dz_range

    def lifted(x, y, z_fallback=None):
        """(N, 3) points on the DEM surface, slightly lifted, in the shifted frame.

        Points off the reconstructed DEM are dropped, unless `z_fallback` (e.g. the
        trajectory's own altitude) is given for them.
        """
        x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        z = _dem_z_at(x, y, x_c, y_c, Z)
        if z_fallback is not None:
            z = np.where(np.isfinite(z), z, np.atleast_1d(z_fallback).astype(float))
        z = z + lift
        keep = np.isfinite(z)
        return np.column_stack([x[keep] - x0, y[keep] - y0, z[keep]])

    plotter = pv.Plotter(off_screen=not show, window_size=(1600, 1000))
    plotter.set_background('white')
    plotter.add_mesh(grid, scalars=column, cmap='RdYlGn_r', clim=clim,
                     nan_color='lightgrey', nan_opacity=1.0, smooth_shading=True,
                     scalar_bar_args=dict(title=column, vertical=True, color='black',
                                          position_x=0.88, position_y=0.2, height=0.6))

    # Outline of the region the trajectories really cover; everything else outside it was
    # extrapolated from the nearest covered cell.
    if mode == 'extrapolate':
        outline = covered & ~binary_erosion(covered)
        if outline.any():
            Xa, Ya = np.meshgrid(x_c, y_c)
            plotter.add_points(lifted(Xa[outline], Ya[outline]), color='black', point_size=2,
                               label='Trajectory corridor')

    # A sample of the trajectories: all of them would turn into a solid band.
    step = max(1, len(listDf) // MAX_TRAJECTORIES_DRAWN)
    for n, df in enumerate(listDf[::step]):
        pts = lifted(df["Est [m]"].values, df["Nord [m]"].values)
        if len(pts) > 1:
            plotter.add_mesh(pv.lines_from_points(pts), color='#262626', line_width=1.5,
                             label=f'Trajectories (1 in {step})' if n == 0 else None)

    markers = []
    if len(gates):
        markers.append((gates["Est [m]"].values, gates["Nord [m]"].values, None, 'purple', 14, 'Gates'))
    start, end = listDf[0].iloc[0], listDf[0].iloc[-1]
    q = "Quota Orto. [m]"
    markers += [(start["Est [m]"], start["Nord [m]"], start[q], 'lime', 20, 'Start'),
                (end["Est [m]"], end["Nord [m]"], end[q], 'cyan', 20, 'End'),
                (*max_xy, 'magenta', 24, f'Max {column.lower()}')]
    for x, y, z, color, size, label in markers:
        pts = lifted(x, y, z)
        if len(pts):
            plotter.add_points(pts, color=color, point_size=size,
                               render_points_as_spheres=True, label=label)

    plotter.add_legend(bcolor='white', border=True, size=(0.2, 0.2), loc='upper left')
    plotter.add_text(f"{column} over the reconstructed slope"
                     + (" (outside the corridor = extrapolated)" if mode == 'extrapolate' else ""),
                     position='upper_edge', font_size=12, color='black')
    plotter.show_bounds(xtitle=f"Est - {x0:.0f} [m]", ytitle=f"Nord - {y0:.0f} [m]",
                        ztitle="Quota [m]", color='black', location='outer')
    plotter.set_scale(zscale=z_exag)

    # Camera on the downhill side, looking up the slope from 50 degrees above the horizon,
    # like the matplotlib figure.
    d = np.asarray(desc_vect[:2], dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    plotter.view_vector((d[0], d[1], np.tan(np.radians(50))), viewup=(0, 0, 1))
    plotter.reset_camera()

    if show:
        print("[Info] Opening the pyvista window (close it to continue)...")
    plotter.show(title=column, screenshot=out, auto_close=True)
    print(f"3D surface saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="3D hazard heatmap from a hazard_coefficients.csv")
    parser.add_argument("csv", help="Path to a hazard_coefficients.csv")
    parser.add_argument("--column", default="Hazard Coefficient",
                        help="CSV column to plot: 'Hazard Coefficient' (default), 'Simulation Coefficient', "
                             "'Bump Coefficient', 'Inclination Coefficient', 'Snow Depth Coefficient', "
                             "or 'all' for one figure per coefficient column.")
    parser.add_argument("--path_to_las", default=DEFAULT_LAS, help="Surface .las used to rebuild the terrain.")
    parser.add_argument("--gates", default=None, help="Optional gates CSV to draw on the surface.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: <column>_surface_3d.png next to the CSV).")
    parser.add_argument("--dem_res", type=float, default=1.0, help="DEM cell size [m].")
    parser.add_argument("--hazard_fill", choices=("extrapolate", "masked"), default="extrapolate",
                        help="'masked' colours only cells crossed by a trajectory.")
    parser.add_argument("--fill_radius", type=float, default=30.0,
                        help="Max extrapolation distance [m] in 'extrapolate' mode.")
    parser.add_argument("--z_exag", type=float, default=1.0, help="Vertical exaggeration.")
    parser.add_argument("--show", action='store_true', help="Open the interactive pyvista viewer (the PNG is saved when it is closed).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    coeff_cols = [c for c in df.columns if c.endswith("Coefficient")]
    if args.column == "all":
        columns = coeff_cols
        if args.out:
            parser.error("--out can only be used with a single --column")
    elif args.column in df.columns:
        columns = [args.column]
    else:
        parser.error(f"column '{args.column}' not in CSV. Available: {coeff_cols}")

    # Rebuild the per-trajectory frames the plot overlays expect.
    listDf = [g.sort_values("Point Index") for _, g in df.groupby("Trajectory Index", sort=True)]

    # Terrain, descent direction and gates do not depend on the column: compute once.
    print(f"Rebuilding the terrain from {args.path_to_las} at {args.dem_res} m ...")
    dem_x, dem_y, dem_z, _ = build_dem(args.path_to_las, res=args.dem_res)
    dem_z = fill_dem_holes(dem_z)
    desc_vect = get_descent_direction(args.path_to_las)
    gates = load_gates(args.gates)

    for column in columns:
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                       column.lower().replace(" ", "_") + "_surface_3d.png")

        river, rel, minX, minY, sx, sy = build_river_matrix(df, column)

        max_row = df[column].values.argmax()
        max_xy = (df["Est [m]"].values[max_row], df["Nord [m]"].values[max_row],
                  df["Quota Orto. [m]"].values[max_row])

        haz_grid, covered = hazard_on_dem(river, rel, dem_x, dem_y, minX, minY, sx, sy, dem_z,
                                          mode=args.hazard_fill, fill_radius=args.fill_radius)

        render_pyvista(dem_x, dem_y, dem_z, haz_grid, covered, listDf, gates, max_xy, desc_vect,
                       column, out, mode=args.hazard_fill, z_exag=args.z_exag, show=args.show)


if __name__ == "__main__":
    main()
