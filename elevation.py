#!/usr/bin/env python3
# Python 3.9 or higher required to run this program.

# pylint: disable=missing-docstring

import argparse
import json
import time
from pathlib import Path
from typing import List, NamedTuple, Optional

import pandas as pd
import pygmt
import numpy as np
from tqdm.auto import tqdm


class ProgramOptions(NamedTuple):
    input_path: Path
    output_path: Path


def get_polygons(geo):
    if geo['type'] == 'Polygon':
        return [geo['coordinates']]
    if geo['type'] == 'MultiPolygon':
        return geo['coordinates']
    if geo['type'] == 'GeometryCollection':
        return [x['coordinates'] for x in geo['geometries'] if x['type'] == 'Polygon']
    raise ValueError('Invalid geometry type: ' + geo['type'])


def create_elevation_data(options: ProgramOptions):
    data = pd.read_csv(options.input_path)

    if options.output_path.exists():
        output_data = pd.read_csv(options.output_path)
        current_values = list(output_data['elevation'])
    else:
        current_values = None

    geo_data = list(data['.geo'].apply(json.loads))
    geo_polygons = list(map(get_polygons, geo_data))

    def get_coordinates(polygons):
        lon = []
        lat = []
        for polygon in polygons:
            for part in polygon:
                lon.extend(x[0] for x in part)
                lat.extend(x[1] for x in part)
        return lat, lon

    def get_latitude(polygons) -> float:
        lat, lon = get_coordinates(polygons)
        return (np.max(lat) + np.min(lat)) / 2

    def get_longitude(polygons) -> float:
        lat, lon = get_coordinates(polygons)
        return (np.max(lon) + np.min(lon)) / 2

    relief_cache = {}
    #relief = pygmt.datasets.load_earth_relief('01s', region=[30, 70, 48, 62], registration='gridline')

    def get_elevation(i, polygons) -> float:
        if current_values and not np.isnan(current_values[i]):
            return current_values[i]

        lat = get_latitude(polygons)
        lon = get_longitude(polygons)
        lat_i = int(lat)
        lon_i = int(lon)
        relief = relief_cache.get((lon_i, lat_i), None)
        if relief is None:
            if len(relief_cache) == 25:
                return None
            relief = pygmt.datasets.load_earth_relief('01s', region=[lon_i, lon_i + 1, lat_i, lat_i + 1], registration='gridline')
            relief_cache[(lon_i, lat_i)] = relief
        return relief.sel(lon=lon, lat=lat, method='nearest').item()

    elevation = list(map(get_elevation, range(len(geo_polygons)), tqdm(geo_polygons)))

    results = data[['id']].copy()
    results['elevation'] = elevation
    results.to_csv(options.output_path, index=False, encoding='ascii', line_terminator='\n')


def parse_command_line_options() -> ProgramOptions:
    parser = argparse.ArgumentParser(
        description='Create elevation data for the Innopolis contest (https://lk.hacks-ai.ru/758467/champ).',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i,--input-path', dest='input_path', type=Path, required=True,
                        help='path to the input csv-file')

    parser.add_argument('-o,--output-path', dest='output_path', type=Path, required=True,
                        help='path to the output csv-file')

    args = parser.parse_args()

    if not args.input_path.exists():
        parser.error(f'argument --input-path: path "{args.input_path}" not found')

    return ProgramOptions(**vars(args))


def main():
    options = parse_command_line_options()

    start_time = time.perf_counter()
    create_elevation_data(options)
    print(f'Total time: {time.perf_counter() - start_time} sec')


if __name__ == '__main__':
    main()
