""" 
    This file contains a simple experiment run
"""
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\X\\Pycharms\\milad\\en2-drone-charging', 'C:/Users/X/Pycharms/milad/en2-drone-charging'])
sys.path.extend(['/root/redflags-honza',])

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from typing import Optional

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

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

from world import WORLD, ENVIRONMENT  # This import should be first
from components.drone_state import DroneState
from utils.visualizers import Visualizer
from utils import plots
from utils.average_log import AverageLog

from ml_deeco.estimators import ConstantEstimator, NeuralNetworkEstimator
from ml_deeco.simulation import run_experiment, SIMULATION_GLOBALS
from ml_deeco.utils import setVerboseLevel, verbosePrint, Log

from drone_charging_example.components.option_enums import SetState, ChargerOption, SimulationType, RedflagsEnum
from shutil import disk_usage


class Run:
    def run(self, args):
        """
        Runs `args.trains` times _iteration_ of [`args.number` times _simulation_ + 1 training].
        """

        # Fix random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)

        if not DISABLE_TF:
            tf.random.set_seed(args.seed)

            # Set number of threads
            tf.config.threading.set_inter_op_parallelism_threads(args.threads)
            tf.config.threading.set_intra_op_parallelism_threads(args.threads)

        yamlObject = self.loadConfig(args)

        folder, yamlFileName = self.prepareFoldersForResults(args)
        estWaitingFolder = f"{folder}/{args.waiting_estimation}"

        averageLog, totalLog, qualityLog = self.createLogs()
        visualizer: Optional[Visualizer] = None

        waitingTimeEstimator = self.createEstimators(args, folder, estWaitingFolder) #TODO this might be disabled
        WORLD.initEstimators()
        identificatiable_name: Optional[str] = None

        def prepareSimulation(i, s):
            """Prepares the _Simulation_ (formerly known as _Run_)."""
            components, ensembles = WORLD.reset(args)
            if args.animation:
                nonlocal visualizer
                visualizer = Visualizer(WORLD)
                visualizer.drawFields()
            return components, ensembles

        def stepCallback(components, materializedEnsembles, step, damage, crop, dead_drones, total_drones):
            """Collect statistics after one _Step_ of the _Simulation_."""
            # print("qualityLog", step)
            qualityLog.register([step, damage, crop, dead_drones, total_drones])
            # def g():
            #     import traceback
            #     for line in traceback.format_stack():
            #         print(line.strip())
            # g()
            # for chargerIndex in range(len(WORLD.chargers)):
            #     charger = WORLD.chargers[chargerIndex]
            #     accepted = set(charger.acceptedDrones)
            #     waiting = set(charger.waitingDrones)
            #     potential = set(charger.potentialDrones)
            #     WORLD.chargerLogs[chargerIndex].register([
            #         len(charger.chargingDrones),
            #         len(accepted),
            #         len(waiting - accepted),
            #         len(potential - waiting - accepted),
            #     ])

            if args.animation:
                visualizer.drawComponents(step + 1, f"{folder}/animations/{yamlFileName}")

        def simulationCallback(components, ensembles, t, i):
            """Collect statistics after each _Simulation_ is done."""
            totalLog.register(self.collectStatistics(t, i))
            # WORLD.chargerLog.export(f"{folder}/charger_logs/{yamlFileName}_{t + 1}_{i + 1}.csv")

            if args.animation:
                verbosePrint(f"Saving animation...", 3)
                nonlocal identificatiable_name
                # visualizer.createAnimation(f"{folder}/animations/final_{yamlFileName}_{t + 1}_{i + 1}.gif", True)
                # visualizer.createAnimation(f"{common_file_name}.gif", True)
                try:
                    visualizer.createAnimation(f"{folder}/animations/{identificatiable_name}.gif", True)
                    verbosePrint(f"Animation saved.", 3)
                except Exception:
                    print(f"ERROR saving animation")

            if args.chart:
                verbosePrint(f"Saving charger plot...", 3)
                plots.createChargerPlot(
                    WORLD.chargerLogs,
                    f"{folder}/charger_logs/{yamlFileName}_{str(t + 1)}_{str(i + 1)}",
                    f"World: {yamlFileName}\nEstimator: {waitingTimeEstimator.estimatorName}\n Run: {i + 1} in training {t + 1}\nCharger Queues")
                verbosePrint(f"Charger plot saved.", 3)

        def iterationCallback(t):
            """Aggregate statistics from all _Simulations_ in one _Iteration_."""

            # calculate the average rate
            averageLog.register(totalLog.average(t * args.number, (t + 1) * args.number))

            for estimator in SIMULATION_GLOBALS.estimators:
                estimator.saveModel(t + 1)

        def check_free_space_for_animation(args):
            if args.animation:
                mb_limit = 0
                # mb_limit = 1000
                # mb_limit = 25000
                save_animation = disk_usage('/dev/shm/').free / 1024 ** 2 > mb_limit
                if not save_animation:
                    print('not enough space for animation', file=sys.stderr)
                    args.animation = False

        which_simulation = str(args.which_simulation).split('.')[1]
        csv_file_name = f"{folder}/{str(args.set_state)}/{which_simulation}/{args.drones}_{yamlFileName}/" \
                           f"{str(args.redflags)}_{args.charger_option}_{args.hysteresis}.csv"
        identificatiable_name = f"{args.drones}__{yamlFileName}__{str(args.set_state)}__{which_simulation}__{str(args.redflags)}__{args.charger_option}__{args.hysteresis}"
        args.save_files_stump = csv_file_name[:-4]

        check_free_space_for_animation(args)

        if args.max_steps != -1:
            ENVIRONMENT.maxSteps = args.max_steps

        output = run_experiment(args.train, args.number, ENVIRONMENT.maxSteps, prepareSimulation,
                       iterationCallback=None, simulationCallback=simulationCallback, stepCallback=stepCallback,
                       args=args)

        os.makedirs(os.path.dirname(csv_file_name), exist_ok=True)
        totalLog.export(csv_file_name)
        qualityLog.export(csv_file_name[:-4] + 'qualiy_log.csv')
        print(f"saving {csv_file_name}")
        # averageLog.export(f"{folder}/{yamlFileName}_{args.waiting_estimation}_average.csv")

        if args.chart:
            plots.createLogPlot(
                totalLog,
                averageLog,
                f"{folder}/{yamlFileName}_{args.waiting_estimation}.png",
                f"World: {yamlFileName}\nEstimator: {waitingTimeEstimator.estimatorName}",
                (args.number, args.train)
            )
        return output

    def loadConfig(self, args):
        # load config from yaml
        yamlFile = open(args.input, 'r')
        yamlObject = load(yamlFile, Loader=Loader)
        if yamlObject is None:
            print(f"File: {args.input} is not valid", file=sys.stderr)
            exit(1)

        # print(f"load config: {args.subfolder=} file {args.input=} {args.birds=} {yamlObject['birds']=}", file=sys.stderr)

        if args.drones > -1:
            yamlObject['drones'] = args.drones
        else:
            args.drones = yamlObject['drones']
            # print(f"debug - loaded config: {args.drones=}", file=sys.stderr)
        if args.drones < 1:
            print(f"At least one drone required: {args.drones}", file=sys.stderr)
            exit(1)
        if int(args.drones) < 1:
            print(f"At least one drone required: {args.drones}\n{args.input=}", file=sys.stderr)
            exit(1)

        if args.birds > -1:
            yamlObject['birds'] = args.birds
            birdsCheck = int(yamlObject['birds'])
        else:
            args.birds = yamlObject['birds']
            birdsCheck = int(yamlObject['birds'])

        # yamlObject['maxSteps']=int(args.timesteps)
        if 'chargerCapacity' not in yamlObject:
            yamlObject['chargerCapacity'] = 1

        if args.no_charger_capacity_increase:
            # this causes change in behavior leading to smoother graphs (but requires more hand-tuning)
            yamlObject['chargerCapacity'] = yamlObject['chargerCapacity']
        else:
            yamlObject['chargerCapacity'] = max(yamlObject['chargerCapacity'], self.findChargerCapacity(yamlObject))

        yamlObject['totalAvailableChargingEnergy'] = min(
            yamlObject['chargerCapacity'] * len(yamlObject['chargers']) * yamlObject['chargingRate'],
            yamlObject['totalAvailableChargingEnergy'])


        ENVIRONMENT.loadConfig(yamlObject)

        return yamlObject

    def findChargerCapacity(self, yamlObject):
        margin = 1.3
        chargers = len(yamlObject['chargers'])
        drones = yamlObject['drones']

        c1 = yamlObject['chargingRate']
        c2 = yamlObject['droneMovingEnergyConsumption']

        return math.ceil(
            (margin * drones * c2) / ((chargers * c1) + (chargers * margin * c2))
        )


    def createLogs(self):
        totalLog = AverageLog([
            'Active Drones',
            'Total Damage',
            'Alive Drone Rate',
            'Damage Rate',
            'Charger Capacity',
            'Train',
            'Run',
        ])
        averageLog = AverageLog([
            'Active Drones',
            'Total Damage',
            'Alive Drone Rate',
            'Damage Rate',
            'Charger Capacity',
            'Train',
            'Average Run',
        ])
        qualityLog = AverageLog([
            'Step',
            'Damage',
            'TotalCrop',
            'DeadDrones',
            'TotalDrones'
        ])
        return averageLog, totalLog, qualityLog



    def prepareFoldersForResults(self, args):
        # prepare folder structure for results
        yamlFileName = os.path.splitext(os.path.basename(args.input))[0]

        if args.no_charger_capacity_increase:
            yamlFileName += 'Q'

        folder = f"{args.output}/results_new/results"
        if args.load_model:
            folder = f"{folder}_{args.load_model}"

        subfolder = args.subfolder
        ch = int(args.chargers_fixed)
        folder = f"{folder}-{args.seed}-{ch}/{subfolder}"  # which_simulation is probably better fit

        print(f"SAVING RESULTS to the: {folder}")
        print("\n"*5)

        if not os.path.exists(f"{folder}/animations"):
            os.makedirs(f"{folder}/animations", exist_ok=True)  # without true race condition
        if not os.path.exists(f"{folder}/charger_logs"):
            os.makedirs(f"{folder}/charger_logs", exist_ok=True)
        return folder, yamlFileName


    def createEstimators(self, args, folder, estWaitingFolder):
        # create the estimators
        commonArgs = {
            "accumulateData": args.accumulate_data,
            "saveCharts": args.chart,
            "testSplit": args.test_split,
        }
        waitingTimeEstimatorArgs = {
            "outputFolder": estWaitingFolder,
            "name": "Waiting Time",
        }
        if args.waiting_estimation == "baseline":
            waitingTimeEstimator = ConstantEstimator(args.baseline, **waitingTimeEstimatorArgs, **commonArgs)
        else:
            waitingTimeEstimator = NeuralNetworkEstimator(
                args.hidden_layers,
                fit_params={
                    "batch_size": 256,
                },
                **waitingTimeEstimatorArgs,
                **commonArgs,
            )
            # if args.load != "":
            #     waitingTimeEstimator.loadModel(args.load)

        WORLD.waitingTimeEstimator = waitingTimeEstimator
        return waitingTimeEstimator


    def collectStatistics(self, train, iteration):
        MAXDRONES = ENVIRONMENT.droneCount if ENVIRONMENT.droneCount > 0 else 1
        MAXDAMAGE = sum([field.allCrops for field in WORLD.fields])

        return [
            len([drone for drone in WORLD.drones if drone.state != DroneState.TERMINATED]),
            sum([field.damage for field in WORLD.fields]),
            len([drone for drone in WORLD.drones if drone.state != DroneState.TERMINATED]) / MAXDRONES,  # rate
            sum([field.damage for field in WORLD.fields]) / MAXDAMAGE,  # rage
            ENVIRONMENT.chargerCapacity,
            train + 1,
            iteration + 1,
        ]


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
    parser.add_argument('-s', '--seed', type=int, help='Random seed.', required=False, default=None)
    parser.add_argument('-b', '--baseline', type=int, help='Constant for baseline.', required=False, default=0)

    parser.add_argument('-l', '--load_model', type=str, help='Load the model from a file.', required=False, default="")

    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    parser.add_argument('-r', '--subfolder', type=str, help='Subfolder for test - changes the used redflags', required=True, default="local_only")

    parser.add_argument('-H', '--hysteresis', type=str, help='a0.4-3, f0.4-3, b0.4-3 - average/forward/backward leaning; number of remembered states', default='')

    # parser.add_argument('-N', '--materialize_ensembles', action='store_false', default=True, help='toggles materialization of the ensembles.')
    parser.add_argument('-N', '--materialize_ensembles', action='store_false', default=False, help='toggles materialization of the ensembles.')
    # parser.add_argument('-F', '--dont_use_redflags', action='store_false')

    parser.add_argument('-W', '--which_simulation', type=str, help='baseline/nn/random_behavior/.../argmax/greedy/slowPoints/slowProb/slowBoth', required=False, default="greedy_ensemble_priority")#default="greedy_drone")
    parser.add_argument('-S', '--set_state', type=str, help='do_nothing/use_set_state_minimal', required=False, default="use_set_state_minimal")
    # parser.add_argument('-S', '--set_state', type=str, help='', required=False, default="set_wish")
    parser.add_argument('-C', '--charger_option', type=str, help='defined in file option_enums  percentage,in_use,full,random', required=False, default="full")
    parser.add_argument('-R', '--redflags', type=str, help='defined in file option_enums', required=False, default="answer")
    parser.add_argument('-X', '--drones', type=int, help='number of drones, if no set, it loads from yaml file.', required=False, default=-1)
    parser.add_argument('-M', '--max_steps', type=int, help='number of total_steps', required=False, default=-1)

    # parser.add_argument('-E', '--which_ensembles', type=str, help='field+original/another_charging', required=False, default="field+another_charging")
    # parser.add_argument('-E', '--which_ensembles', type=str, help='field+original/another_charging', required=False, default="another_charging")
    parser.add_argument('-E', '--which_ensembles', type=str, help='field+minimal/another_charging', required=False, default="field+minimal_charging")
    parser.add_argument('-B', '--chargers_fixed', help='Are charging drones fixed in ens composition? standart - no, they can be changed', action='store_true', default=False)

    # parser.add_argument('-P', '--save_predictions_and_results', action='store_true', default=False, help='toggles saving of each step to a file')
    parser.add_argument('-Q', '--no_charger_capacity_increase', action='store_true', default=False, help='toggles increase in charger capacity according to the number of drones, default is change the capacity')

    # parser.add_argument('-C', '--charger_option', type=int, help='defined in file option_enums 0,1,2,3 : percentage,in_use,full,random', required=False, default=2)


    # args = parser.parse_args('/root/redflags-honza/drone_charging_example/experiments/4drones.yaml --set_state do_nothing --which_simulation random_behavior --dont_use_redflags --hysteresis 0 --charger_option percentage --subfolder local_only'.split())
    args = parser.parse_args()
    args.which_simulation = SimulationType[args.which_simulation]
    args.set_state = SetState[args.set_state]
    args.charger_option = ChargerOption[args.charger_option]
    args.redflags = RedflagsEnum[args.redflags]

    if args.output is None:
        if sys.platform == 'linux':
            args.output = '/dev/shm'
        else:
            args.output = 'outputs'

    number = args.number
    setVerboseLevel(args.verbose)

    if number <= 0:
        raise argparse.ArgumentTypeError(f"{number} is an invalid positive int value")

    output = Run().run(args)
    print(args)

    # following prints are allowing the xargs to show progress with removal of all extra info
    # print("seed:", args.seed, "redflag:", args.redflags,  "state:", args.set_state, "simulation:", args.which_simulation, "hys:", args.hysteresis, "charger_options:", args.charger_option, "subfolder:", args.subfolder, file=sys.stderr)
    output = "Damage: {0}/{1} - Dead drones: {2}/{3}".format(*output)
    print("seed:", args.seed, "drones:", args.drones, "nn", args.load_model, output, 'set_state:', args.set_state, 'which_simulation:', args.which_simulation, 'redflags:', args.redflags, 'hysteresis:', args.hysteresis, 'charger_option:', args.charger_option, 'subfolder:', args.subfolder, args.input.split('/')[-1], file=sys.stderr)


if __name__ == "__main__":
    main()
