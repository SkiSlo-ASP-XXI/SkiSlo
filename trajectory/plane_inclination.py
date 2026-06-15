"""
Convert the tangent derivative from tangent_derivative.py into the signed
inclination angle (in degrees) along the trajectory.

The tangent derivative D = grad(z) . t_hat is the rise per metre travelled
along the trajectory's horizontal direction, so the signed slope angle of the
path is:

        theta = arctan( D )        [degrees, negative when descending]

where D == tangent_derivative is stored by tangent_derivative.py in its
detailed CSV.

Input
-----
The CSV written by tangent_derivative.py (column tangent_derivative).

Output
------
  * <out>.npy   the raw signed-angle array [degrees] (NaN where the
                plane was not evaluated)
  * <out>.csv   easting, northing, tangent_derivative, inclination_deg
"""

import argparse
import os
import sys

import numpy as np

# Allow running as a plain script: make the project root importable so the
# absolute import below works regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skier_model_py.physical_model import esegui_simulazione


def tangent_angle_deg(tangent_derivative):
    """Signed slope angle [deg] along the trajectory: arctan(D)."""
    return np.degrees(np.arctan(tangent_derivative))


def main():
    parser = argparse.ArgumentParser(
        description="Signed slope angle (deg) along the trajectory: arctan(D)."
    )
    parser.add_argument(
        "--csv",
        default="outputs/tangent_derivative_F_tr1_d1.csv",
        help="CSV produced by tangent_derivative.py (with dz_dx, dz_dy columns).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output basename (no extension). "
        "Default: outputs/plane_inclination_<csv-name>.",
    )
    args = parser.parse_args()

    table = np.genfromtxt(args.csv, delimiter=",", names=True)
    easting = table["easting"]
    northing = table["northing"]
    height = table["height"]
    deriv = table["tangent_derivative"]

    angle = tangent_angle_deg(deriv)

    if args.out is None:
        name = os.path.splitext(os.path.basename(args.csv))[0]
        args.out = os.path.join("outputs", f"tangent_angle_{name}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    valid = np.isfinite(angle)
    print(
        f"{valid.sum()}/{len(angle)} points evaluated; "
        f"angle range [{np.nanmin(angle):.3f}, {np.nanmax(angle):.3f}] deg, "
        f"mean {np.nanmean(angle):.3f} deg"
    )

    npy_path = args.out + ".npy"
    np.save(npy_path, angle)
    print(f"Saved angle array: {npy_path}")

    csv_path = args.out + ".csv"
    out_table = np.column_stack([easting, northing, deriv, angle])
    np.savetxt(
        csv_path,
        out_table,
        delimiter=",",
        header="easting,northing,tangent_derivative,inclination_deg",
        comments="",
    )
    print(f"Saved table:       {csv_path}")

    # Run the skier physics along the same trajectory, forcing the per-point
    # slope angle (alfa) to the inclination we just computed from the surface.
    #
    # Sign convention: our `angle = arctan(D)` is the signed slope along travel,
    # so it is NEGATIVE while descending. esegui_simulazione instead defines its
    # slope as alpha = -arctan2(dz, horizontal) (descent -> POSITIVE), where a
    # positive alpha drives the skier forward via Fs = m*g*sin(alpha). Passing
    # our angle un-negated makes gravity brake instead of drive, so the skier
    # stops after a couple of metres. Negate to match the model's convention.
    print("Running esegui_simulazione with alfa = computed inclination ...")
    risultato = esegui_simulazione(easting, northing, height, alfa=-angle, plot=True)

    return risultato



if __name__ == "__main__":
    main()