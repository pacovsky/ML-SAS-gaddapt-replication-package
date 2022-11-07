# % #  %  # py #thon.ipython

import pandas as pd
import wandb
import itertools
import pprint
import os
# os.environ["WANDB_MODE"] = "offline"
import json

os.environ["WANDB_DIR"] = "/dev/shm/wandb"
#%%
def get_df():
    def _get_df(path):
        return pd.read_csv(path, dtype={'redflags': str})

    df = None
    name = 'shell_output.csv'
    name = 'shell_output_not_random.csv'
    name = 'shell_output_not_random_compact_nn.csv'

    try:
        path = r'C:\Users\X\Pycharms\milad\en2-drone-charging\external_utils' + f'\\{name}'
        df = _get_df(path)
    except:
        path = r'/root/redflags-honza/external_utils' + f'/{name}'
        df = _get_df(path)

    if df is None:
        raise FileNotFoundError("shell_output does not exist on the location")

    return df

def redflags_group(df):
    df_red = df.groupby('redflags')
    header = 'seed,ones,redflags,survived,eaten,HasDoneNothingWantsToCharge,HighEnergyWantsToCharge,ChargeOnVeryLowEnergy,LowEnergyFlyToField,FalseDetectionOfTerminated,DroneOnEmptyField,HighEnergyIdleDronesVeryFarFromBirds,WrongDroneCountField,TooMuchDronesAssignToTheCharger'

    e = df_red['eaten']
    s = df_red['survived']
    df_es = pd.concat([e.min(), e.mean(), e.max(), s.min(), s.mean(), s.max()], axis=1)
    df_es.columns = ["eaten_min", "eaten_mean", "eaten_max", "survived_min", "survived_mean", "survived_max"]


def log_wandb(df):
    for row in df.itertuples():
        d = row._asdict()
        index = d.pop('Index')
        log = {'survived': d.pop('survived'),
               'eaten': d.pop('eaten')
               }
        run = wandb.init(sync_tensorboard=False, project="eaten_survived_compact_nn", entity='deepcharles',
                         reinit=True,  # Allow multiple wandb.init() calls in the same process
                         mode="offline",
                         config=d
                         )
        # wandb.config.update(d)
        wandb.log(log)
        run.finish()

        print(index)

def for_parallel_wandb_row(d):
    # d = row._asdict()
    index = d.pop('Index')
    log = {'survived': d.pop('survived'),
           'eaten': d.pop('eaten')
           }
    run = wandb.init(sync_tensorboard=False, project="eaten_survived", entity='deepcharles',
                     reinit=True,  # Allow multiple wandb.init() calls in the same process
                     mode="offline",
                     config=d
                     )
    # wandb.config.update(d)
    wandb.log(log)
    run.finish()
    print(index)

def log_wandb_parallel(df):
    from multiprocessing import Pool
    rows = (row._asdict() for row in df.itertuples())


    pool = Pool(processes=70)
    pool.map(for_parallel_wandb_row, rows)

def log_wandb_to_bash(df):
    for row in df.itertuples():
        d = row._asdict()
        print(json.dumps(d), end='\0')

def print_options_setup():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):



def filter_columns_redflags(df, indexes):
    # one e.g. triplet / pair ...
    return df.iloc[:, [*indexes]].product(axis=1)

def get_combs_names(df, indexes):
    # index_names = df.keys()[[*indexes]]
    # return index_names
    return [str(i) for i in indexes]


class ComputationCorelationAllWithAll:
    def corr_all_with_all(self, df):
        def get_redundant_pairs(df):
            '''Get diagonal and lower triangular pairs of correlation matrix'''
            pairs_to_drop = set()
            cols = df.columns
            for i in range(0, df.shape[1]):
                for j in range(0, i + 1):
                    pairs_to_drop.add((cols[i], cols[j]))
            return pairs_to_drop


        def get_top_abs_correlations(df):
            au_corr = df.corr().unstack()
            labels_to_drop = get_redundant_pairs(df)
            au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
            return au_corr


        def get_dumm(df):
            dumm = pd.get_dummies(df['redflags'])
            all = pd.concat((dumm, df), axis=1)
            all = all.iloc[:, :-9]
            return all

        print("Top Absolute Correlations")
        dumm = get_dumm(df)
        print(get_top_abs_correlations(dumm, 300))


def print_cross_analysis(df):
    df_se = df.iloc[:, 3:5]  # dataframe_survived_eaten
    df_rf = df.iloc[:, 5:]  # dataframe_redflags

    df.iloc[:, 3:5].corrwith(df.iloc[:, 5] + df.iloc[:, 6])
    ones = 2
    dfs = []
    combs = []

    for ones in range(1, 10):
        for n_tuple in itertools.combinations(range(9), ones):
            combs.append(get_combs_names(df_rf, n_tuple))
            n_redflags = filter_columns_redflags(df_rf, n_tuple)
            x = df_se.corrwith(n_redflags)
            dfs.append(x)

    c = [','.join(i) for i in combs]
    twos = pd.concat(dfs, axis=1)
    twos.columns = c
    sorted_twos = twos.unstack().sort_values(ascending=True, key=abs)
    small = sorted_twos[abs(sorted_twos)> 0.2]

    print(small.to_string())
    pprint.pprint(dict(zip(range(10), df_rf.keys())))


if __name__ == '__main__':
    df = get_df()
    # print_cross_analysis(df)
    # log_wandb(df.head(3))
    # log_wandb_to_bash(df.head(3))
    log_wandb_to_bash(df)
