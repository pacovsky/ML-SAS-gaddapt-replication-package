#%%
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from itertools import combinations
import time
import os
import sys

def load_a_model(model_filepath):
    # this way can be load both models and checkpoints
    # '/mnt/ensml/logs/wandb/run-20210409_213939-ct54h5u0/files/model-best.h5'
    # '/mnt/ensml/logs/wandb/run-20210413_122620-yb9xt0i9/files/checkpoints/'
    try:
        try:
            model = tf.keras.models.load_model(model_filepath, custom_objects={'StochasticDepth': tfa.layers.StochasticDepth})
            return model
        except OSError:
            print("OSError - not found?:", model_filepath)
            return None
        except:
            print("Error - not found?:", model_filepath)
            return None
    except:
        return None


def get_model_paths():
    #dir=/dev/shm/nn-logs; ls -1 $dir | cut -d. -f2 | sort -n | tail -n 5 | xargs -i find $dir -name "*0.{}.h5"
    return [
        '/dev/shm/nn-logs/2022_03_31_09-48-32-0.9761964678764343.h5',
        '/dev/shm/nn-logs/2022_03_31_12-36-53-0.9762064814567566.h5',
        '/dev/shm/nn-logs/2022_03_30_12-53-09-0.9763191938400269.h5',
        '/dev/shm/nn-logs/2022_03_30_01-20-38-0.9763468503952026.h5',
        '/dev/shm/nn-logs/2022_03_30_00-50-41-0.9765072464942932.h5',
        '/dev/shm/nn-logs/2022_03_30_13-54-40-0.9765704870223999.h5',
        '/dev/shm/nn-logs/2022_03_29_23-23-11-0.9767979383468628.h5',
        '/dev/shm/nn-logs/2022_03_30_00-24-19-0.9771632552146912.h5',
        '/dev/shm/nn-logs/2022_03_30_03-11-01-0.9774134755134583.h5',
        '/dev/shm/nn-logs/2022_03_30_03-33-46-0.9779280424118042.h5',
    ]



def get_models(paths):
    # is the path absolute?
    assert paths[0][0] == '/'

    models_or_nones = [load_a_model(x) for x in paths]
    loaded_models = list(filter(lambda a: a is not None, models_or_nones))
    return loaded_models


def get_models_without_eval(top=0):
    """paths may be sorted in order of accuracy (increasing)"""
    paths = get_model_paths()[-top:]
    models = get_models(paths)
    return models


def get_sorted_models_with_evaluation(X, y, last=0, paths=None):
    """:param last: get only last x models
    :param paths: if given uses that path"""
    model_paths = paths if paths else get_model_paths()
    model_paths = model_paths[-last:]
    loaded_models = get_models(model_paths)

    sorted_models, evaluations = evaluate_models_sort(X, y, loaded_models, model_paths)
    loaded_models = [x[1] for x in sorted_models]
    print(sorted_models)
    return loaded_models, evaluations


def evaluate_models_sort(x, y, models, paths, batch_size=3000):
    """returns models and result of evaluation sorted by the accuracy"""
    top = []
    evaluations = []
    print("models evaluation:")
    for (model, path) in zip(models, paths):
        print(path)
        loss, evaluation = model.evaluate(x, y, batch_size)
        top.append((evaluation, model, path))
        evaluations.append(evaluation)
        print(evaluation)
        print(path)

        print("+++=+++\n")

    top = sorted(top, key=lambda l: -l[0])
    return top, evaluations


def create_ensemble(input_width, ensemble_models, optim=None, loss=None, metrics=None):
    input_layer = tf.keras.layers.Input(input_width, name="ensemble_input_layer")
    m = [ModelWrapper(one_model)(input_layer) for one_model in ensemble_models]
    assert len(m) > 1, (f"number of models:{len(m)} (should be > 1 )")

    if type(m[0]) == list:
        #  we are in multiple output network - we need to avg outputs that belong together
        avg = [tf.keras.layers.Average()(part_output) for part_output in zip(*m)]
    else:
        avg = tf.keras.layers.Average()(m)

    created_ensemble = tf.keras.Model([input_layer], avg)
    created_ensemble.compile(
        optimizer=optim if optim else tf.optimizers.Adam(),
        loss=loss if loss else tf.losses.CategoricalCrossentropy(),
        metrics=metrics if metrics else [tf.metrics.CategoricalAccuracy()]
    )

    return created_ensemble

class ModelWrapper(tf.keras.Model):
    """wrapper to ensure each model has its own namespace"""
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def call(self, input_tensors, training=None, mask=None):
        return self.model(input_tensors)


def load_test_data():
    # ls -dR1 /dev/shm/datasets/*/*
    file = [
        '/dev/shm/datasets/testing/16667_AIO.npz',
        '/dev/shm/datasets/testing/167_AIO.npz',
        '/dev/shm/datasets/validation/16667_AIO.npz',
        '/dev/shm/datasets/validation/167_AIO.npz'
    ][0]
    # ][2]
    # load precompiled file
    d = np.load(file, mmap_mode='r')
    X, y = d['data'], d['labels']
    return X, y




#%%
def get_model_combinations(model_list, stop=None, start=2):
    """param: stop denotes max size of n-tuples"""
    def get_combinations(array, stop=None, start=2):
        if not stop:
            stop = len(array)

        combs = []
        for i in range(start, stop + 1):
            combs.extend([comb for comb in combinations(array, i)])

        return combs
    stop = stop if stop else len(model_list)
    # combos = get_combinations(range(len(model_list))[::-1], stop, start)
    combos = [
        [(9, 8), (9, 7), (9, 6), (8, 7), (8, 6), (7, 6)],  # 2
        [(9, 8, 4, 2),  (9, 8, 6, 2),  (9, 8, 7),  (9, 8, 7, 5),  (9, 8, 7, 4),  (9, 8, 4),  (9, 8, 4, 1),  (9, 8, 6),  (9, 8, 2),  (9, 8, 5, 4),  (9, 8, 1)], #3,4
        [(9, 8, 5, 4, 2), (9, 8, 6, 4, 2), (9, 8, 7, 6, 4), (9, 8, 7, 5, 2), (9, 8, 7, 4, 2), (9, 8, 7, 5, 4), (9, 8, 7, 2, 0), (9, 8, 7, 6, 5), (9, 8, 7, 4, 0), (9, 8, 6, 5, 2), (9, 8, 6, 4, 1), (9, 8, 7, 4, 3), (9, 8, 4, 2, 1), (9, 8, 6, 5, 4), (9, 8, 7, 5, 0), (9, 8, 7, 2, 1)], #5
    ]
    combos = [val for sublist in combos for val in sublist]
    print("combos to do: ", combos)
    models = np.array(model_list)
    return [(models[[x]], x) for x in combos]



#%%
if __name__ == "__main__":
    def do_evaluation(models, evaluations):
        ensemble = create_ensemble(109, models)
        loss, metrics = ensemble.evaluate(X, y, batch_size=30000)
        evaluation = metrics
        evaluations.append(evaluation)
        return evaluations

    parser = argparse.ArgumentParser()
    parser.add_argument("--top", default=0, type=int, help="number of models to use; 0 to use all paths")
    parser.add_argument("--evaluate", default=False, type=bool, help="Evaluate combining models before use")
    parser.add_argument("--combine", default=True, type=bool, help="Evaluate combininations")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    top = args.top

    X, y = load_test_data()

    evaluations = []
    if args.evaluate:
        models, evaluations = get_sorted_models_with_evaluation(X, y, top)
    else:
        models = get_models_without_eval(top)

    print(len(models), " of models to try to combine")
    if args.combine:
        evaluations = []
        desc = []
        for m, description in get_model_combinations(models, 5,5): #(models, 4,3)
            print(description)
            desc.append(description)
            do_evaluation(m, evaluations)

        results = sorted(list(zip(evaluations, desc)), key=lambda x: x[0], reverse=True)
        print(results)
    else:
        do_evaluation(models, evaluations)
        print(evaluations)

#
#0.9783033728599548 - 21
#0.9790992736816406 - 3
#0.9790194630622864 - 4
#0.9790244698524475 - 2
#0.9790219664573669 - 5
