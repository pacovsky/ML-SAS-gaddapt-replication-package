"""
    This file contains a simple experiment run
"""
import sys;
import time

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\X\\Pycharms\\milad\\en2-drone-charging', 'C:/Users/X/Pycharms/milad/en2-drone-charging'])
sys.path.extend(['/root/redflags-honza',])

from typing import Optional

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import os
import argparse
from datetime import datetime
import random
import numpy as np
import math

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
DISABLE_TF = True
if not DISABLE_TF:
    import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='Process YAML source file (S) and run the simulation (N) Times with Model M.')
    parser.add_argument('input', type=str, help='YAML address to be run.')
    parser.add_argument('-x', '--birds', type=int, help='number of birds, if no set, it loads from yaml file.', required=False, default=-1)
    parser.add_argument('-n', '--number', type=int, help='the number of simulation runs per training.', required=False, default="1")
    parser.add_argument('-t', '--train', type=int, help='the number of trainings to be performed.', required=False, default="1")
    parser.add_argument('-o', '--output', type=str, help='the output folder', required=False, default=None)
    parser.add_argument('-v', '--verbose', type=int, help='the verboseness between 0 and 4.', required=False, default="0")
    parser.add_argument('-a', '--animation', action='store_true', default=False,
                        help='toggles saving the final results as a GIF animation.')
    parser.add_argument('-c', '--chart', action='store_true', default=False, help='toggles saving and showing the charts.')
    parser.add_argument('-w', '--waiting_estimation', type=str,
                        choices=["baseline", "neural_network"],
                        help='The estimation model to be used for predicting charger waiting time.', required=False,
                        default="neural_network")
    parser.add_argument('-d', '--accumulate_data', action='store_true', default=False,
                        help='False = use only training data from last iteration.\nTrue = accumulate training data from all previous iterations.')
    parser.add_argument('--test_split', type=float, help='Number of records used for evaluation.', required=False, default=0.2)
    parser.add_argument('--hidden_layers', nargs="+", type=int, default=[16, 16], help='Number of neurons in hidden layers.')
    parser.add_argument('-s', '--seed', type=int, help='Random seed.', required=False, default=43)
    parser.add_argument('-b', '--baseline', type=int, help='Constant for baseline.', required=False, default=0)
    parser.add_argument('-l', '--load', type=str, help='Load the model from a file.', required=False, default="")

    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    parser.add_argument('-r', '--subfolder', type=str, help='Subfolder for test - chabnges the used redflags', required=False, default=None)
    args = parser.parse_args()

    if args.output is None:
        if sys.platform == 'linux':
            args.output = '/dev/shm/compact_input/'
        else:
            args.output = 'outputs'

    number = args.number

    if number <= 0:
        raise argparse.ArgumentTypeError(f"{number} is an invalid positive int value")

    print(args)


if __name__ == "__main__":
    print("running main")
    main()
    time.sleep(5)
    if random.random() > 0.7:
        raise RuntimeError("trow")
    else:
        print("done", file=sys.stderr)

