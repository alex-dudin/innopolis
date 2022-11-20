#!/usr/bin/env python3
# Python 3.9 or higher required to run this program.

# pylint: disable=missing-docstring

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import List, NamedTuple, Optional

import pandas as pd
import numpy as np


logger = logging.getLogger('innopolis')


class ProgramOptions(NamedTuple):
    input_path: List[Path]
    output_path: Path
    weights: Optional[List[float]] = None


TARGET_COUNT = 7


def create_submission(options: ProgramOptions):
    logger.info('Read predicts...')
    data = [pd.read_csv(path) for path in options.input_path]

    logger.info('Blend predicts...')
    predicts = []
    for target_index in range(TARGET_COUNT):
        sum_predict = None
        for df_num, df in enumerate(data):
            predict = df[f'p{target_index}']
            if options.weights:
                predict *= options.weights[df_num]

            if sum_predict is None:
                sum_predict = predict
            else:
                sum_predict += predict

        predicts.append(sum_predict)

    submission = data[0][['id']].copy()
    submission['crop'] = np.argmax(predicts, axis=0)
    
    logger.info('Create submission...')
    submission.to_csv(options.output_path, index=False, encoding='ascii', line_terminator='\n')


def configure_logging():
    formatter = logging.Formatter('%(asctime)s [%(levelname)5s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    logger.setLevel(1) # min level


def parse_command_line_options() -> ProgramOptions:
    parser = argparse.ArgumentParser(
        description='Create submission for the Innopolis contest (https://lk.hacks-ai.ru/758467/champ).',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i,--input-path', dest='input_path', action='append', type=Path,
                        help='path to the input csv-file that contains predicts')

    parser.add_argument('-o,--output-path', dest='output_path', type=Path, required=True,
                        help='path to the output csv-file')

    parser.add_argument('-w,--weights', dest='weights', type=str,
                        help='comma-separated list of weights for all targets')

    args = parser.parse_args()

    for path in args.input_path:
        if not path.exists():
            parser.error(f'argument --input-path: path "{path}" not found')

    if args.weights:
        args.weights = list(map(float, args.weights.split(',')))
        if len(args.weights) != len(args.input_path):
            parser.error(f'argument --weights: invalid weights count {args.weights}')

    return ProgramOptions(**vars(args))


def main():
    options = parse_command_line_options()

    configure_logging()

    start_time = time.perf_counter()
    create_submission(options)
    logger.info(f'Total time: {time.perf_counter() - start_time} sec')


if __name__ == '__main__':
    main()
