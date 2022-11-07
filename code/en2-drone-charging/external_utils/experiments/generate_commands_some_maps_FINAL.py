import glob
from itertools import permutations
import re
import tqdm
import os
import time
import pandas as pd
from os.path import exists

def get_files():
    fs = [
        "/root/redflags-honza/drone_charging_example/experiments/10drones.yaml",

        # '/root/redflags-honza/drone_charging_example/experiments/FINAL/1tiny.yaml',
        # # '/root/redflags-honza/drone_charging_example/experiments/FINAL/5dice-2-1000epoch.yaml',
        '/root/redflags-honza/drone_charging_example/experiments/FINAL/5dice-3.yaml',
        '/root/redflags-honza/drone_charging_example/experiments/FINAL/fill_one_drone.yaml',
        '/root/redflags-honza/drone_charging_example/experiments/FINAL/orig.yaml',
        # # '/root/redflags-honza/drone_charging_example/experiments/FINAL/orig-100birds.yaml',
        '/root/redflags-honza/drone_charging_example/experiments/FINAL/real-life.yaml',
        '/root/redflags-honza/drone_charging_example/experiments/FINAL/U-1-ch.yaml',
   ]
    for f in fs:
        yield f


def get_hysteresis_old():
    hysteresis = [""]
    for string in 'fbFB':#"fbaFB":
        rates = [.1, .2, 0.5, 0.8, 0.9]
        for rate in rates:
            for length in [3, 5, 9]:
                # skip different rate averages (no effect)
                if string != "a" or rate == rates[0]:
                    hysteresis.append(f"{string}{rate}-{length}")
    return hysteresis

def get_hysteresis():
    HHHH = ['a', 'c', 'e', 'j', 'm', 'p', 's', 'v','f', 'h', 'g', 'k', 'n', 'q', 't', 'w','b', 'd', 'i', 'l', 'o', 'r', 'u', 'x',]
    HHHH += [c for c in "".join(HHHH).upper()]
     # HHHH = ['i', 'l', 'o', 'r', 'u', 'x']
    HHHH_END = [
        '0.15-5',
        '0.10-5',
        '0.15-10',
        '0.2-10',
        '0.5-10'
    ]

    import itertools
    hysteresis = list(''.join(x) for x in itertools.product(HHHH, HHHH_END))
    return hysteresis

def read_info_from_file(path):
    with open(path, 'r') as f:
        f.seek(84)  # after the header

        # active_drones, total_damage, alive_drone_rate, damage_rate, charger_capacity, train, run
        line = f.readline()

        # we might care for charger_capacity
        # but never for train, run
        vals = line.split(',', 5)[:-1]
        types = [int, int, float, float, int]
        parsed = [t(v) for v, t in zip(vals, types)]
        # active_drones, total_damage, alive_drone_rate, damage_rate, charger_capacity = parsed
        return parsed


def get_info_from_path(path):
    # csv_file_name = f"{folder}/{str(args.set_state)}/{which_simulation}/{yamlFileName}/" \
    #                     f"{str(args.redflags)}_{args.charger_option}_{args.hysteresis}.csv"

    parts = path.split('/')[4:]
    # ['results-3', 'local_only', 'SetState.do_nothing', 'nn', '44drones', 'RedflagsEnum.none_3_0.csv']
    # print(parts, path)
    folder, subfolder, set_state, which_simulation, file, rest = parts
    stump, seed, chargers_fixed = folder.split('-', 2)

    nnet = "default"  # "C1"  # which is the default one

    try:
        nnet = stump.split('_', 1)[1]
    except IndexError:
        pass

    redflags, charger_option, hysteresis = rest.split('_')
    hysteresis = hysteresis[:-4]  # get rid of csv
    if not hysteresis:
        hysteresis = "none"

    drones, file = file.split('_', 1)

    return [seed, chargers_fixed, nnet, subfolder, set_state, which_simulation, drones, file, redflags, charger_option, hysteresis]


def geather(csv, search_path, red_flags_expanded=False, filter=None):
    # head
    path_info_header = ['seed', 'ch_fixed', 'nnet', 'subfolder', 'set_state', 'which_simulation', 'drones', 'file', 'redflags', 'charger_option', 'hysteresis']
    file_info_header = ['active_drones', 'total_damage', 'alive_drone_rate', 'damage_rate', 'charger_capacity']
    RF = path_info_header.index('subfolder')

    if red_flags_expanded:
        rf_columns = ['HighEnergyLevelChargeStopper',
            'HighEnergyWantsToCharge',
            'ChargeOnVeryLowEnergy',
            'LowEnergyFlyToField',
            'DroneOnEmptyField',
            'HighEnergyIdleDronesVeryFarFromBirds'
        ]
        path_info_header = path_info_header[:RF] + rf_columns + path_info_header[RF+1:]

    header = path_info_header + file_info_header

    series = []
    for path in tqdm.tqdm(glob.glob(search_path, recursive=True)):
        if filter is not None and filter(path):
            continue
        if 'qualiy_log' in path:
            continue

        path_info = get_info_from_path(path)
        file_info = read_info_from_file(path)
        infos = path_info + file_info

        if red_flags_expanded:
            integers = re.findall('([01])', infos[RF])

            # contained a digit
            if integers:
                # if len(integers) != 8:
                #     print(f"error on {integers} (has length of {len(integers)}, in ", path)
                #     continue # skips assert
                assert len(integers) == 8, f"ints: '{integers}' have length {len(integers)}"
                binary = integers[:-2]
            else:
                binary = ['0'] * 6

            infos = infos[:RF] + binary + infos[RF+1:]


        series.append(infos)

    df = pd.DataFrame(series, columns=header)
    paths = ['/dev/shm/', '/mnt/ensml/', '/tmp/']
    # csv = '_RL_hysteresis_big.CSV'

    for path in paths:
        if exists(path + csv):
            age_in_seconds = time.time() - os.path.getmtime(path + csv)
            print('writing', path + csv)
            df.to_csv(path + csv, index=False)
            #
            # hour = 3600 * 15
            # if age_in_seconds < hour:
            #     print(f'do you want to override file: {path + csv}?')
            #     inp = 'yes'
            #     # inp = 'no'
            #     # inp = input(f"do you want to override file: {path + csv}? Write 'yes'")
            #     if inp == 'yes':
            #         df.to_csv(path + csv, index=False)
            #         print('writing', path + csv)
            #     else:
            #         continue
            # else:
            #     print("Protection activated")
            #     print(f"NOT SAVING {path}, (age in sec was: {age_in_seconds})")
            # raise FileNotFoundError("file exists! -> forcing fail")
        else:
            df.to_csv(path + csv, index=False)
            print('writing', path+csv)
    print("name of the file (if saved)", csv)
    return [path + csv for path in paths]


############ filters for main #############
def filter_results(path):
    if '-wrong-assigns' in path:
        return True
    if 'slow' in path:
        return True

    return 'qualiy_log' in path


def create_seeds(m, M, lis):
    ou = []
    for j in range(m, M):
        ou.extend([i + f' --seed {j} ' for i in lis])
    return ou


def create_run_files(save_path, seed, seed2, drones):
    # maps = set(generator())
    maps = set(final_gen(drones))
    maps = sorted(maps)
    maps = create_seeds(seed, seed2, maps)

    #    if 'reorder' todo problems
    # give_2_end = "greedy_ensemble2"
    # s = [i for i in maps if give_2_end not in i] #NAMING IS WRONG
    # bez = [i for i in maps if give_2_end in i]
    # wih = create_seeds(1000, 1005, s)
    # wihou = create_seeds(1000, 1005, bez)
    # wih2 = create_seeds(1005, 1010, s)
    # wihou2 = create_seeds(1005, 1010, bez)
    # maps = wih + wihou + wih2 + wihou2
    # print(wih[0])



    # todo subfolder 111111 has implementation error, one_ch shows another - check the capacity
    with open(save_path, 'w') as file:
        for ccount, g in enumerate(maps):
            file.write(g + '\n')
            print(g)
    print(g)
    print(ccount)


def collect_results():
    d = {
        # 'everything': '*',
        # '1000': '100[0-9]',
        # '1100': '11[0-9][0-9]',
        '1200': '12[0-9][0-9]',
        'final': '[0-9]',
        'hys_d_vs_b': '89[0-9]',
        'hys_d_vs_b2': '88[0-4]',
    }
    for n, p, in d.items():
        if n == 'everything':
            on = f'ens_1000_everything.csv'
            on = f'ens_c_1000_everything.csv'
        else:
            on = f'ens_{n}.csv'
            on2 = f'ens_c_{n}.csv'

        sp = f'/dev/shm/results_new/results*-{p}-[01]/**/*.csv'

        # sp = f'/dev/shm/results_new/results*-{p}/**/*.csv'

        names = geather(on, sp, red_flags_expanded=True, filter=filter_results)
        # names = geather(on2, sp, red_flags_expanded=False, filter=filter_results)
        print(p)


def generator():
    pyt = '/root/EEEE/VENV_DIR/bin/python'
    script = '/root/redflags-honza/drone_charging_example/run.py'
    dir = '/root/redflags-honza/drone_charging_example/experiments'

    # charger_option = ['percentage', 'in_use', 'full', 'random']
    # states = ['do_nothing', 'set_wish', 'use_set_state', 'use_set_state_minimal', 'do_not_enter_the_processing']
    # rfs = ['answer', 'argmax', 'none']
    # sims = ['argmax', 'baseline', 'betterbaseline', 'greedy_ensemble', 'greedy_drone', greedy_drone2, greedy_drone_ens]
    # neural_nets = ['C1', 'C2', 'C1_1', 'C1_2', 'C1_3', '256', '256B', 'deep']

    composiiton_simulations = {
        'baseline_with_composiion',
        'greedy_drone',  # OK
        'greedy_drone2',
        'greedy_drone_ens',
        'greedy_ensemble_priority',  # ok is the partial sets with
        'greedy_ensemble',  # normal
        'greedy_ensemble2',  # with partial sets
        'greedy_ensemble1_priority',  # extra field priority without partial sets
    }
    # hysteresis = get_hysteresis()
    charger_option = ['full']
    states = ['use_set_state_minimal']
    rfs = ['answer']
    neural_nets = ['256B']
    subfolder = ['11111100', '00000000']

    selected_drones = [10, 20, 40, 50]
    # short_long_drones = list(range(3, 50))

    short_long_drones = selected_drones + list(range(3, 11)) + list(range(11, 30, 2)) + list(range(30, 70, 5)) + list(range(70, 150, 15))
    print(short_long_drones)
    # short_long_drones = list(sorted(set(short_long_drones)))
    long_drones = list(range(2, 100)) + [110,120,130,140,150]
    long_drones = list(range(2, 75))  # list(range(2, 101))
    short_long_drones = long_drones
    # long_drones = list(sorted(set(long_drones) - set(short_long_drones)))

    selected_drones.append("")
    short_long_drones.append("")



    print("output")
    # short_long_drones = (set(short_long_drones).symmetric_difference(long_drones))

    # # REMOVE 10,20,30,40,50
    # x = set(range(1, 30))
    # x = x - set(selected_drones[1:])
    # x = sorted(x)
    # selected_drones = x

    # hysteresis = ["", 'a0.1-5', 'c0.1-5',]# 'B0.15-5', 'D0.15-5']
    # hysteresis = ["", 'a0.1-5', 'c0.1-5', 'B0.15-5', 'D0.15-5']
    hysteresis = ["", 'a0.1-5', 'B0.15-5']
    # hysteresis = ['B0.15-5', 'D0.15-5']
    # hysteresis = ["", 'a0.1-3', 'a0.1-5', 'b0.5-9', 'b0.9-9']
    # hysteresis = ["", 'a0.1-5', 'B0.5-9', 'B0.9-9', 'f0.8-9']

    sims = [
            'betterbaseline', # this should be equal to base-argmax with rf ==0
            # 'baseline_with_rf', WRONG! chjo
            'baseline_with_rf_argmax',
            'baseline_with_composiion',
            'argmax',
            'greedy_drone',  #OK
            'greedy_drone2',
            'greedy_drone_ens',
            'greedy_ensemble_priority',  # ok is the partial sets with

            'greedy_ensemble', #normal
            'greedy_ensemble2',#with partial sets
            'greedy_ensemble1_priority', #extra field priority without partial sets
            ]
    just_full = sims[4:]

    # TODO remove this
    sims = ['greedy_ensemble_priority', 'greedy_drone']
    selected_drones = []  # batch 2

    short_long_drones = list(set(range(2, 30)) - set([4,7,11,14,17,22,29,37,58])) + [33,39,41,43,48]
    short_long_drones = [4, 7, 11, 14, 17, 22, 29, 37, 58]
    subfolder = ['11111100']


    hysteresis = get_hysteresis()

    # hysteresis = [
    #     'B0.15-5',
    #     'd0.15-5',
    #     # '',
    #     'd0.10-5',
    #     'd0.15-10',
    #     'd0.2-10',
    #     'd0.5-10',
    # ]

    # sims = ['argmax']
    last_line = ''
    for file in get_files():
      if '/10drones.yaml' not in file:
          continue
      for drone in short_long_drones if '/10drones.yaml' in file else selected_drones:
        for sim in sims:
          for s in subfolder if '/10drones.yaml' in file else ['11111100']:
            for ch_o in charger_option:
              for rf in rfs:
                for hys in hysteresis:
                  for state in states:
                    for net in neural_nets:
                      for chf in ["--chargers_fixed",]:
                      # for chf in ["--chargers_fixed", '']:
                          for _ in ['']:
                            if sim not in composiiton_simulations:
                              continue
                            if sim in just_full:
                              if '/10drones.yaml' not in file:
                                  if not hys or '000' in s:
                                      continue
                          ss = f"--set_state {state}" if state else ""
                          ws = f"--which_simulation {sim}" if sim else ""
                          fr = f"--redflags {rf}" if rf else ""
                          hy = f"--hysteresis {hys}" if hys else ""
                          ch = f"--charger_option {ch_o}" if ch_o else ""
                          su = f"--subfolder {s}" if s else ""
                          dr = f" --drones {drone}" if drone else ""
                          nn = f" --load_model {net}" if net else ""
                          # suffix = f"-M 10 -a -X 8 -x 2"
                          suffix = ""
                          if 'baseline' in sim and sim not in ['baseline_with_rf', 'baseline_with_rf_argmax', 'baseline_with_composiion']:  # baseline does not subject to changes all combinations yield the same result
                              if '1' in su or hy: # skip RF, hys
                                  continue
                              # else:
                              #     chf = ''
                              #     yield fr'{script} \"{file}\" {ss} {ws} {fr} {hy} {ch} {su} {dr} {nn} {chf} {suffix}'
                          else:
                            yield fr'{script} \"{file}\" {ss} {ws} {fr} {hy} {ch} {su} {dr} {nn} {chf} {suffix}'
                            # pass

def final_gen(selected_drones):
    def combos():
        # # m1, m2, argmax, baseline, bl+rf,
        # # returns method, hyst, rf and pre
        yield 'greedy_drone', True, True, True
        yield 'greedy_ensemble_priority', True, True, True
        # yield 'argmax', False, False, False
        # yield 'betterbaseline', False, False, False
        # yield 'baseline_with_rf_argmax', False, True, False
        # yield 'baseline_with_composiion', False, False, True
        # # yield 'baseline_with_composiion', True, True, True

        # yield 'greedy_ensemble_obey_partial', True, True, True
        # yield 'greedy_ensemble_obey', True, True, True


    hY = 'B0.15-5'
    hY = 'd0.15-5'
    hYs = [#'d0.15-5',
                       # 'B0.15-5',
                       # '',
                        'd0.10-5',
                        'd0.15-10',
                        'd0.2-10',
                        'd0.5-10',
                       ]
    script = '/root/redflags-honza/drone_charging_example/run.py'

    for file in get_files():
        for sim, hys, rf, pre in combos():
            for hY in get_hysteresis():

                ss = f"--set_state use_set_state_minimal"
                ws = f"--which_simulation {sim}"
                su = f"--subfolder {'11111100' if rf else '00000000'}"
                hy = f"--hysteresis {hY}" if hys and hY else ""
                ch = f"--charger_option full"
                fr = f"--redflags answer"
                # dr = f" --drones {drone}" if drone else ""
                nn = f" --load_model 256B"
                chf = '--chargers_fixed' if pre else ''
                yield fr'{script} \"{file}\" {ss} {ws} {fr} {hy} {ch} {su} {nn} {chf} '

if __name__ == '__main__':
    save_path = "/root/redflags-honza/external_utils/experiments/final_commands_d_hys.run"

    # create_run_files(save_path, 882, 885, None)
    create_run_files(save_path, 873, 879, None)

    collect_results()
#


