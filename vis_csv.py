"""
Visualize a CloudCompare CSV point cloud (`data/snow_depth/Nuvola_differenza_neve.csv`).

Renders a two-panel figure: a 3D scatter plus a top-down XY map. The CSV header
is commented out by CloudCompare (`//X,Y,Z,Relative height`), and in these
snow-difference clouds Z *is* the difference - Z and `Relative height` are the
same column - so the scalar is labelled and coloured as a depth difference
rather than as an elevation.

The colour scale spans a robust percentile range: a handful of gross outliers
(tens of metres) would otherwise flatten the whole scale around the ~1.5 m of
real signal. When the clipped range straddles zero the map is diverging and
centred on 0, the no-change level; otherwise it is sequential.

An optional `--open3d` flag opens the interactive Open3D viewer. open3d is only
installed in `.venv` / `.lasreading`, so it is imported lazily and the script
degrades with a message when it is missing.

Usage
-----
    python vis_csv.py                          # 2-panel figure -> vis_csv.png
    python vis_csv.py --color z                # colour by the raw Z column
    python vis_csv.py --clip 0 100             # no percentile clipping
    python vis_csv.py --max-points 0           # plot every point (slow, 10M)
    python vis_csv.py --no-show                # save the PNG only (headless)
    .venv/bin/python vis_csv.py --open3d       # interactive 3D viewer

Output
------
    vis_csv.png     two-panel figure written to the repo root
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

# ---------------- SETTINGS ----------------
ROOT            = Path(__file__).resolve().parent
SNOW_DIR        = ROOT / "data" / "snow_depth"
DEFAULT_CSV     = SNOW_DIR / "Nuvola_differenza_neve.csv"
DEFAULT_PNG     = "vis_csv.png"
DEFAULT_MAX_PTS = 150_000          # visual downsample; 0 = plot everything
DIVERGING_CMAP  = "RdBu_r"         # used when the scalar straddles 0
SEQUENTIAL_CMAP = "turbo"          # used when it does not
DEFAULT_CLIP    = (2.0, 98.0)      # robust percentile range for the colour scale
# ------------------------------------------


def resolve_csv_path(arg):
    """
    Resolve the CSV path, tolerating a missing `data/snow_depth/` component.

    Parameters
    ----------
    arg : str or Path
        Path as given on the command line. Relative paths are resolved against
        the repo root, not the current working directory.

    Returns
    -------
    Path
        An existing CSV file.
    """
    p = Path(arg)
    if not p.is_absolute():
        p = ROOT / p

    if p.is_file():
        return p

    # e.g. `data/Nuvola_differenza_neve.csv` -> `data/snow_depth/Nuvola_...csv`
    fallback = SNOW_DIR / p.name
    if fallback.is_file():
        print(f"[Info] {p} not found, using {fallback} instead.")
        return fallback

    available = sorted(SNOW_DIR.glob("*.csv")) if SNOW_DIR.is_dir() else []
    listing = "\n".join(f"  - {f.relative_to(ROOT)}" for f in available) or "  (none)"
    raise SystemExit(f"[Error] CSV file not found: {p}\nAvailable clouds:\n{listing}")


def load_csv(path):
    """
    Read a CloudCompare CSV export.

    X/Y stay float64 - they are UTM metres, and float32 only carries ~7
    significant digits, which is coarser than a decimetre at 3.3e5 m. Z and the
    relative height are read as float32, which halves the memory on a 10M-row
    file.

    Returns
    -------
    P : (N, 3) float64 ndarray
        XYZ coordinates in the file's CRS.
    depth : (N,) float32 ndarray or None
        The `Relative height` column, or None when the file has no such column.
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    # CloudCompare comments out the header line: `//X,Y,Z,...`
    names = [c.lstrip("/").strip() for c in header]
    if len(names) < 3:
        raise SystemExit(f"[Error] {path.name} has fewer than 3 columns: {names}")

    xyz = names[:3]
    depth_col = next((n for n in names[3:] if "height" in n.lower()), None)

    usecols = xyz + ([depth_col] if depth_col else [])
    dtype = {xyz[0]: np.float64, xyz[1]: np.float64, xyz[2]: np.float32}
    if depth_col:
        dtype[depth_col] = np.float32

    print(f"[Info] Reading {path.name} ({path.stat().st_size / 1e6:.0f} MB), "
          "this takes a few seconds...")
    df = pd.read_csv(path, header=0, names=names, usecols=usecols, dtype=dtype)

    P = np.column_stack([df[xyz[0]].to_numpy(),
                         df[xyz[1]].to_numpy(),
                         df[xyz[2]].to_numpy().astype(np.float64)])
    depth = df[depth_col].to_numpy() if depth_col else None

    print(f"[OK] Loaded {path.name}: {P.shape[0]:,} points, columns {names}")
    print(f"     X: {P[:, 0].min():.2f} .. {P[:, 0].max():.2f} m")
    print(f"     Y: {P[:, 1].min():.2f} .. {P[:, 1].max():.2f} m")
    print(f"     Z: {P[:, 2].min():.2f} .. {P[:, 2].max():.2f} m")
    if depth is not None:
        same = np.allclose(P[:, 2], depth, atol=1e-4)
        print(f"     {depth_col}: {depth.min():.2f} .. {depth.max():.2f} m"
              f"{' (identical to Z)' if same else ''}")

    return P, depth


def pick_colors(scalar, cmap_name, clip):
    """
    Map the scalar to per-point RGB.

    The normalisation spans the `clip` percentile range rather than the raw
    min/max, so that isolated outliers do not compress the colour scale. A range
    that straddles 0 is made symmetric about it and drawn with a diverging map,
    which puts "no change" at the neutral centre colour.

    Returns
    -------
    colors : (N, 3) float ndarray
        Per-point RGB in [0, 1].
    norm : Normalize
        The normalisation used, so the colorbar can match the points.
    cmap : Colormap
    """
    if clip[1] > clip[0]:
        lo, hi = (float(v) for v in np.percentile(scalar, clip))
    else:
        lo, hi = float(scalar.min()), float(scalar.max())

    diverging = lo < 0.0 < hi
    if diverging:
        half = max(abs(lo), abs(hi))
        lo, hi = -half, half

    if cmap_name == "auto":
        cmap_name = DIVERGING_CMAP if diverging else SEQUENTIAL_CMAP

    if clip[1] > clip[0]:
        outside = int(((scalar < lo) | (scalar > hi)).sum())
        print(f"[Info] Colour scale clipped to the {clip[0]:g}-{clip[1]:g} percentile "
              f"-> [{lo:.2f}, {hi:.2f}] m; {outside:,} points saturate "
              "(use --clip 0 100 to disable).")

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=lo, vmax=hi)
    return cmap(norm(scalar))[:, :3], norm, cmap


def subsample(P, colors, max_points, seed=0):
    """Randomly thin the cloud for plotting only. Reproducible via `seed`."""
    n = P.shape[0]
    if max_points <= 0 or n <= max_points:
        return P, colors, n

    idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
    idx.sort()
    print(f"[Info] Subsampled {n:,} -> {max_points:,} points for plotting "
          f"(use --max-points 0 to plot all).")
    return P[idx], colors[idx], n


def plot_matplotlib(P, colors, norm, label, cmap, n_total, args, zlim=None):
    """
    Draw the 3D scatter + top-down map, save the PNG and optionally show it.

    `zlim` bounds the 3D z-axis; outliers otherwise stretch it so far that the
    real relief collapses onto a single plane. The z axis is then exaggerated
    relative to the true metric aspect, because a decimetre-scale difference
    over a half-kilometre footprint is invisible at 1:1 - the factor is stated
    in the panel title.
    """
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    dx, dy = np.ptp(x), np.ptp(y)
    z_lo, z_hi = zlim if zlim is not None else (float(z.min()), float(z.max()))
    dz = max(z_hi - z_lo, 1e-9)

    # Give the z axis about an eighth of the footprint's visual extent, capped:
    # past ~20x the sheet reads as noise rather than as relief.
    exag = min(20.0, max(1.0, 0.12 * max(dx, dy) / dz))

    # Constrained layout, not tight_layout. Neither engine measures the 3D
    # z-axis label (it is drawn outside the axes rect), so the panels also get
    # an explicit gap to keep it clear of the map's y-axis label.
    fig = plt.figure(figsize=(16, 7), layout="constrained")
    fig.get_layout_engine().set(wspace=0.12)

    # ---------- 1) 3D scatter (coordinates centred: UTM values are huge) ----------
    x0, y0 = float(x.mean()), float(y.mean())
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.scatter(x - x0, y - y0, np.clip(z, z_lo, z_hi),
                 c=colors, s=args.point_size, marker=".", linewidths=0, depthshade=False)
    ax3d.set_xlabel(f"X - {x0:.0f} [m]")
    ax3d.set_ylabel(f"Y - {y0:.0f} [m]")
    ax3d.set_zlabel(label)
    ax3d.set_zlim(z_lo, z_hi)
    # The exaggerated z axis is still short next to the footprint, so the default
    # tick density would overprint the labels.
    ax3d.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax3d.set_box_aspect((max(dx, 1e-9), max(dy, 1e-9), dz * exag))
    ax3d.set_title("3D view" if exag <= 1.0 else f"3D view (Z exaggerated {exag:.0f}x)")

    # ---------- 2) Top-down XY map ----------
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.scatter(x, y, c=colors, s=args.point_size, marker=".", linewidths=0)
    ax2d.set_xlabel("Easting [m]")
    ax2d.set_ylabel("Northing [m]")
    ax2d.set_aspect("equal")
    ax2d.ticklabel_format(style="plain", useOffset=True)
    ax2d.set_title("Top-down view")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax2d, shrink=0.85, extend="both").set_label(label)

    shown = P.shape[0]
    subtitle = f"{shown:,} of {n_total:,} points" if shown != n_total else f"{n_total:,} points"
    fig.suptitle(f"{Path(args.csv).name}  -  {subtitle}, colored by {label}")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    fig.savefig(out, dpi=200, bbox_inches="tight", format="png")
    print(f"[OK] Figure saved to {out}")

    if args.show:
        plt.show()
    plt.close(fig)


def view_open3d(P, colors):
    """Open the interactive Open3D viewer, if open3d is available."""
    try:
        import open3d as o3d
    except ImportError:
        print("[Info] open3d is not installed in this interpreter, skipping the "
              "interactive viewer.\n"
              "       Try:  .venv/bin/python vis_csv.py --open3d")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P - P.mean(axis=0))   # centre: open3d is float32 inside
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64)[:, :3])

    size = float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))) * 0.1
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(size, 1.0))

    print("[Info] Opening the open3d window (close it to continue)...")
    o3d.visualization.draw_geometries([pcd, frame],
                                      window_name="Nuvola_differenza_neve",
                                      width=1600, height=900)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open and plot a CloudCompare CSV point cloud "
                    "(defaults to data/snow_depth/Nuvola_differenza_neve.csv).")

    # paths for loading and saving data (OPTIONAL)
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                        help="path to the CSV file (default: %(default)s)")
    parser.add_argument("--out", default=DEFAULT_PNG,
                        help="output PNG for the matplotlib figure (default: %(default)s)")

    # parameters for plotting (OPTIONAL)
    parser.add_argument("--color", choices=("auto", "depth", "z"), default="auto",
                        help="colour by the relative-height column or by raw Z "
                             "(auto = relative height when present) (default: %(default)s)")
    parser.add_argument("--cmap", default="auto",
                        help=f"matplotlib colormap, auto = {DIVERGING_CMAP} when the range "
                             f"straddles 0 else {SEQUENTIAL_CMAP} (default: %(default)s)")
    parser.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"),
                        default=list(DEFAULT_CLIP),
                        help="percentile range for the colour scale, "
                             "0 100 = raw min/max (default: %(default)s)")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_PTS,
                        help="max points to plot, 0 = all (default: %(default)s)")
    parser.add_argument("--point-size", type=float, default=0.5,
                        help="scatter marker size (default: %(default)s)")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="save the figure without opening a window")
    parser.add_argument("--open3d", action="store_true",
                        help="also open the interactive open3d viewer (needs .venv/.lasreading)")

    return parser.parse_args()


def main():
    args = parse_args()

    args.csv = resolve_csv_path(args.csv)
    P, depth = load_csv(args.csv)

    if args.color in ("auto", "depth") and depth is not None:
        scalar, label = depth, "Snow depth difference [m]"
    else:
        if args.color == "depth":
            print("[Info] --color depth requested but the file has no relative-height "
                  "column; falling back to Z.")
        scalar, label = P[:, 2].astype(np.float32), "Z [m]"

    colors, norm, cmap = pick_colors(scalar, args.cmap, tuple(args.clip))

    Pp, Cp, n_total = subsample(P, colors, args.max_points)

    # Bound the 3D z-axis with the same robust percentiles used for the colours,
    # computed on the full cloud rather than on the subsample.
    clip = tuple(args.clip)
    zlim = ((float(np.percentile(P[:, 2], clip[0])), float(np.percentile(P[:, 2], clip[1])))
            if clip[1] > clip[0] else None)

    # `norm` stays the full-cloud normalisation: the colors were mapped through
    # it, so the colorbar must span it too, not just the subsampled points.
    print("Max Z:", float(P[:, 2].max()), "m")
    print("Min Z:", float(P[:, 2].min()), "m")
    plot_matplotlib(Pp, Cp, norm, label, cmap, n_total, args, zlim=zlim)

    if args.open3d:
        view_open3d(P, colors)


if __name__ == "__main__":
    main()
