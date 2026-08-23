#!/usr/bin/env python3
"""Turn a CloudCompare snow-depth export into the compact `.npz` the demo uploads.

A cloud-to-cloud export of a single slope is a *text* file of several hundred
megabytes — the one this was written against is 532 MB for 10,035,538 points — and
three quarters of that is redundant:

  * `Z` and `Relative height` are the same number twice (verified to 1e-6 on the
    reference file), and the pipeline reads only the second one;
  * the coordinates are printed to 8 decimals, i.e. nanometres of UTM easting;
  * the points sit on a ~0.2 m lattice, so a raster loses nothing real.

So this rasterises: one cell per `--res` metres, depth quantised to the centimetre
as int16, nodata where the survey has no points. On the reference file that is
1.8 MB at the default 0.25 m — 295x smaller — and only ~26% of the bounding box is
occupied (a slope is a corridor, not a rectangle), so the nodata runs compress well.

    python tools/snow_depth_light.py snow_depth.csv snow_depth.npz

Deliberately depends on numpy and pandas only: converting the big file happens on
whatever machine can hold it, which is not necessarily one with the demo's full
environment installed. `pipeline.SnowDepthTree.from_npz` reads what this writes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Depth is stored as int16 centimetres: 1 cm resolution over +/- 327.67 m, against
# a reference file spanning -31.33 .. +28.88 m. The most negative int16 is the hole.
NODATA = np.int16(-32768)
DEPTH_MIN_M, DEPTH_MAX_M = -327.67, 327.67

# Kept in step with `SnowDepthTree._DEPTH_ALIASES` in pipeline.py, which is the
# authority on what a snow-depth file may call its columns.
DEPTH_ALIASES = ("relative height", "relative_height", "depth")


def resolve_columns(columns):
    """The x, y and depth column names, however the exporter spelled them.

    Mirrors `SnowDepthTree._columns` (pipeline.py): case-insensitive, whitespace
    stripped, and a leading `/` removed so CloudCompare's `//X` header matches.
    """
    cols = {str(c).strip().lstrip("/").lower(): c for c in columns}
    x_col, y_col = cols.get("x"), cols.get("y")
    depth_col = next((cols[a] for a in DEPTH_ALIASES if a in cols), None)
    if x_col is None or y_col is None or depth_col is None:
        raise SystemExit(
            f"the snow-depth file needs columns x, y and one of {DEPTH_ALIASES}; "
            f"found {list(columns)}"
        )
    return x_col, y_col, depth_col


def peek_columns(path: Path, sep: str):
    head = pd.read_csv(path, sep=sep, header=0, nrows=5)
    return resolve_columns(head.columns)


def scan_bounds(path: Path, sep: str, x_col, y_col, chunksize: int):
    """First pass: the extent, reading only the two coordinate columns."""
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    total = 0
    for chunk in pd.read_csv(path, sep=sep, header=0, usecols=[x_col, y_col],
                             chunksize=chunksize):
        x = chunk[x_col].to_numpy(dtype=np.float64)
        y = chunk[y_col].to_numpy(dtype=np.float64)
        xmin, xmax = min(xmin, x.min()), max(xmax, x.max())
        ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
        total += len(chunk)
        print(f"\r  bounds: {total:,} points", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if total == 0:
        raise SystemExit(f"{path} has no data rows")
    return xmin, xmax, ymin, ymax, total


def accumulate(path, sep, x_col, y_col, depth_col, chunksize,
               x0, y0, res, nrows, ncols, stat):
    """Second pass: fold every point into its cell.

    Memory is bounded by the grid, never by the cloud — the same `np.bincount`
    trick `build_dem` uses in pipeline.py. At 0.25 m over the reference extent that
    is 6.4 M cells, so ~77 MB of accumulators for a 532 MB input.
    """
    cells = nrows * ncols
    count = np.zeros(cells, dtype=np.int32)
    if stat == "mean":
        acc = np.zeros(cells, dtype=np.float64)
    else:
        acc = np.full(cells, np.inf if stat == "min" else -np.inf, dtype=np.float64)

    total = 0
    for chunk in pd.read_csv(path, sep=sep, header=0,
                             usecols=[x_col, y_col, depth_col], chunksize=chunksize):
        x = chunk[x_col].to_numpy(dtype=np.float64)
        y = chunk[y_col].to_numpy(dtype=np.float64)
        d = chunk[depth_col].to_numpy(dtype=np.float64)

        # Drop rows the exporter left as NaN rather than letting them poison a cell.
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
        if not good.all():
            x, y, d = x[good], y[good], d[good]

        j = np.rint((x - x0) / res).astype(np.int64)
        i = np.rint((y - y0) / res).astype(np.int64)
        np.clip(j, 0, ncols - 1, out=j)
        np.clip(i, 0, nrows - 1, out=i)
        flat = i * ncols + j

        count += np.bincount(flat, minlength=cells).astype(np.int32)
        if stat == "mean":
            acc += np.bincount(flat, weights=d, minlength=cells)
        elif stat == "min":
            np.minimum.at(acc, flat, d)
        else:
            np.maximum.at(acc, flat, d)

        total += len(chunk)
        print(f"\r  binning: {total:,} points", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return acc, count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterise a CloudCompare snow-depth export into a compact .npz.")
    parser.add_argument("src", type=Path, help="the CloudCompare .csv export")
    parser.add_argument("dst", type=Path, help="the .npz to write")
    parser.add_argument("--res", type=float, default=0.25,
                        help="cell size in metres (default 0.25, about the native "
                             "point spacing; 0.5 is ~3.5x smaller again)")
    parser.add_argument("--stat", choices=("mean", "min", "max"), default="mean",
                        help="how to combine the points inside one cell (default mean; "
                             "median is not available because it cannot be computed "
                             "in a single streaming pass)")
    parser.add_argument("--sep", default=",",
                        help="column separator of the export (default ',')")
    parser.add_argument("--chunksize", type=int, default=2_000_000,
                        help="rows per read (default 2,000,000)")
    args = parser.parse_args()

    if args.res <= 0:
        raise SystemExit("--res must be positive")
    res = args.res

    if not args.src.exists():
        raise SystemExit(f"{args.src} does not exist")

    started = time.time()
    src_mb = args.src.stat().st_size / 1e6
    print(f"Reading {args.src} ({src_mb:,.1f} MB)", file=sys.stderr)

    x_col, y_col, depth_col = peek_columns(args.src, args.sep)
    print(f"  columns: x={x_col!r} y={y_col!r} depth={depth_col!r}", file=sys.stderr)

    xmin, xmax, ymin, ymax, total = scan_bounds(
        args.src, args.sep, x_col, y_col, args.chunksize)
    ncols = int(round((xmax - xmin) / res)) + 1
    nrows = int(round((ymax - ymin) / res)) + 1
    print(f"  extent: X {xmin:.2f}..{xmax:.2f} ({xmax - xmin:.1f} m), "
          f"Y {ymin:.2f}..{ymax:.2f} ({ymax - ymin:.1f} m)", file=sys.stderr)
    print(f"  grid: {nrows} x {ncols} = {nrows * ncols:,} cells at {res} m",
          file=sys.stderr)

    acc, count = accumulate(args.src, args.sep, x_col, y_col, depth_col,
                            args.chunksize, xmin, ymin, res, nrows, ncols, args.stat)

    filled = count > 0
    n_filled = int(filled.sum())
    if n_filled == 0:
        raise SystemExit("no cell received a point — check --sep and the header")

    values = np.where(filled, acc / np.maximum(count, 1) if args.stat == "mean" else acc,
                      0.0)
    out_of_range = int(np.count_nonzero(
        filled & ((values < DEPTH_MIN_M) | (values > DEPTH_MAX_M))))
    if out_of_range:
        print(f"  warning: {out_of_range:,} cells clipped to +/-327 m", file=sys.stderr)

    depth_cm = np.full(nrows * ncols, NODATA, dtype=np.int16)
    depth_cm[filled] = np.rint(
        np.clip(values[filled], DEPTH_MIN_M, DEPTH_MAX_M) * 100.0).astype(np.int16)
    depth_cm = depth_cm.reshape(nrows, ncols)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.dst,
        kind="raster_v1",
        res=np.float64(res),
        x0=np.float64(xmin),
        y0=np.float64(ymin),
        depth_cm=depth_cm,
    )

    dst_mb = args.dst.stat().st_size / 1e6
    depths = depth_cm[depth_cm != NODATA].astype(np.float64) / 100.0
    print(f"\nWrote {args.dst} ({dst_mb:,.2f} MB)", file=sys.stderr)
    print(f"  {total:,} points -> {n_filled:,} cells "
          f"({100.0 * n_filled / (nrows * ncols):.1f}% of the grid)", file=sys.stderr)
    print(f"  depth: min {depths.min():.2f} m, median {np.median(depths):.2f} m, "
          f"max {depths.max():.2f} m", file=sys.stderr)
    print(f"  {src_mb / dst_mb:,.0f}x smaller, in {time.time() - started:.1f} s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
