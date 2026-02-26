import argparse
import importlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


GNSS_KEYS = [
    'lat',
    'lon',
    'height_ellipsoid',
    'height_msl',
    'ground_speed',
    'vel_east',
    'vel_north',
    'vel_down',
    'speed_acc',
    'horizontal_acc',
    'vertical_acc',
    'heading_acc',
    'time_acc',
    'timestamp',
    'year',
    'month',
    'day',
    'hour',
    'min',
    'sec',
    'nano',
    'itow',
    'nb_sats',
    'position_dop',
    'heading',
    'fix_type',
    'flags',
    'flags2',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract IMU and GNSS data from an H5 file to JSON.')
    parser.add_argument('--file', required=True, help='Path to input H5 file (dirs + nomefile.h5).')
    parser.add_argument('--skier', required=False, default=None, help='Skier name.')
    parser.add_argument('--track', required=False, default=None, help='Track number/name.')
    parser.add_argument('--run', required=False, default=None, help='Run number.')
    parser.add_argument('--outputPath', default=None,help='Output JSON path. If omitted, saves beside input H5 using *_extracted.json suffix.')

    args = parser.parse_args()
    if not args.file.endswith('.h5'):
        raise ValueError('Input file must have .h5 extension.')
    
    if args.skier is None or args.track is None or args.run is None:
        print("Warning: Skier, track, or run information is missing. Metadata will be taken from the H5 file if possible.")
        basename = Path(args.file).stem
        parts = basename.split('_')
        if len(parts) >= 3:
            args.skier = args.skier or parts[0]
            args.track = args.track or parts[1]
            args.run = args.run or parts[2].split('.')[0]
        else:
            raise ValueError('Input filename does not contain enough parts to infer skier, track, and run. Please provide them as arguments or ensure the filename is in the format skier_track_run.h5.')

    return args



def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value


def normalize_lat_lon(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_f = lat.astype(float)
    lon_f = lon.astype(float)

    if np.nanmax(np.abs(lat_f)) > 90:
        lat_f = lat_f * 1e-7
    if np.nanmax(np.abs(lon_f)) > 180:
        lon_f = lon_f * 1e-7

    return lat_f, lon_f


def wgs84_to_utm32n(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        pyproj = importlib.import_module('pyproj')
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Missing dependency 'pyproj'. Install it with: pip install pyproj"
        ) from exc

    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return np.asarray(easting), np.asarray(northing)


def first_valid_date_component(values: np.ndarray) -> int:
    valid = values[~np.isnan(values.astype(float))]
    if valid.size == 0:
        return -1
    return int(valid[0])


def build_output(input_file: Path, skier: str, track: str, run: str) -> dict:
    with h5py.File(input_file, 'r') as data:
        imu_data = {
            'timestamp': data['sensor/imu/timestamp'][:],
            'acc': data['sensor/imu/acc'][:, :],
            'gyr': data['sensor/imu/gyr'][:, :],
        }

        gnss_data = {key: data[f'sensor/gnss/{key}'][:] for key in GNSS_KEYS}

    lat_deg, lon_deg = normalize_lat_lon(gnss_data['lat'], gnss_data['lon'])
    utm_easting, utm_northing = wgs84_to_utm32n(lat_deg, lon_deg)

    metadata = {
        'skier': skier,
        'track': track,
        'run': run,
        'day': first_valid_date_component(gnss_data['day']),
        'month': first_valid_date_component(gnss_data['month']),
        'year': first_valid_date_component(gnss_data['year']),
    }

    gnss_without_date = {k: v for k, v in gnss_data.items() if k not in {'year', 'month', 'day'}}
    gnss_without_date['utm32n_easting'] = utm_easting
    gnss_without_date['utm32n_northing'] = utm_northing
    
    return {
        'metadata': metadata,
        'imu_data': imu_data,
        'gnss_data': gnss_without_date,
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.file)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    output_path = Path(args.outputPath) if args.outputPath else "./results"

    output = build_output(input_file=input_path, skier=args.skier, track=args.track, run=args.run)

    output_path.parent.mkdir(exist_ok=True)

    with open(output_path / f"{input_path.stem}.json", 'w', encoding='utf-8') as f:
        json.dump(to_serializable(output), f, indent=2, ensure_ascii=False)

    print(f'JSON saved to: {output_path / f"{input_path.stem}.json"}')


if __name__ == '__main__':
    main()