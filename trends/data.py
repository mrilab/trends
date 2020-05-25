from collections import namedtuple
from pandas import (read_csv, concat)

Set = namedtuple("Set", ["training", "test"])
Data = namedtuple("Data", ["X", "y"])

def get_data(data_dir="../data/"):
    loading = read_csv(f"{data_dir}/loading.csv", header=0, index_col=0)
    fnc = read_csv(f"{data_dir}/fnc.csv", header=0, index_col=0)
    X = concat([loading, fnc], axis=1)
    y = read_csv(f"{data_dir}/train_scores.csv", header=0, index_col=0)
    y[y.isna()] = 50
    return Set(Data(X.loc[y.index], y), Data(X.drop(index=y.index), None))
