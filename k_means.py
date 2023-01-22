import random
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt

from cantor_fit import return_curves

SIZE_OF_DF = 500

SIZE_TOTAL = 1000000


def get_all_data():
    locations = np.load("locations.npy")[:SIZE_TOTAL]
    amount_in_pair = np.load("amount_in_pair.npy")[:SIZE_TOTAL]

    dists = locations[1:] - locations[:-1]  # TODO: missing one line
    merged_data = np.hstack([dists.reshape(dists.shape[0], 1)[:-1], amount_in_pair[1:-1]])  # TODO: there are nulls
    df = pd.DataFrame(merged_data).dropna()
    df.columns = ['dist', '0->0', '0->1', '1->0', "1->1"]
    df = df[df["dist"] > 0].dropna()
    df = df[df["dist"] <= 400].dropna()
    return df


def get_params(df):
    df = df.groupby("dist", as_index=False).sum()
    prior_0 = (df['0->0'] + df['0->1'])
    prior_1 = df['1->0'] + df['1->1']
    df['0->0'] /= prior_0
    df['0->1'] /= prior_0
    df['1->0'] /= prior_1
    df['1->1'] /= prior_1
    df = df.dropna()
    return return_curves(df)


def argmin(lst):
    return lst.index(min(lst))


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def get_params_of_group(list_of_dfs):
    if len(list_of_dfs) == 0:
        return lambda x: x * 0
    dfs_merged = pd.concat(list_of_dfs, ignore_index=True, sort=False)
    return get_params(dfs_merged)


def get_dist(functions, df):
    diff1 = (functions[0](df["dist"]) - df["0->0"]).abs().sum()
    diff2 = (functions[1](df["dist"]) - df["0->1"]).abs().sum()
    diff3 = (functions[2](df["dist"]) - df["1->0"]).abs().sum()
    diff4 = (functions[3](df["dist"]) - df["1->1"]).abs().sum()
    return diff1 + diff2 + diff3 + diff4


def merge_groups(groups):
    return [pd.concat(group, ignore_index=True, sort=False) for group in groups]


def k_means(df, k, iters):
    print(len(df))
    dfs = [df.iloc[i * SIZE_OF_DF:(i + 1) * SIZE_OF_DF] for i in range(len(df) // SIZE_OF_DF)]

    groups = partition(dfs, k)

    for _ in range(iters):
        print(_)
        params_for_group = [get_params_of_group(group) for group in groups]

        # clean groups:
        groups = [[] for _ in range(k)]
        count = 0
        for df in dfs:
            count += 1
            print(count)
            best_params_ind = argmin([get_dist(params, df) for params in params_for_group])  # TODO
            groups[best_params_ind].append(df)

    return merge_groups(groups), [get_params_of_group(group) for group in groups]


def run(df,k, num_iter=2):
    groups, functions_for_groups = k_means(df, k, num_iter)
    print(len(groups[0]), len(groups[1]))

    for functions in functions_for_groups:
        fig = px.line(x=range_of_dists, y=[function(range_of_dists) for function in functions],
                     title=f"K means results k={k}")
        newnames = {"wide_variable_0": "0->0",
                    "wide_variable_1": "0->1",
                    "wide_variable_2": "1->0",
                    "wide_variable_3": "1->1"}

        fig.for_each_trace(lambda t: t.update(name=newnames[t.name]))
        fig.show()

if __name__ == '__main__':
    df = get_all_data()
    org_functions = get_params(df)
    range_of_dists = np.linspace(0, 400, 1000)

    for k in [2, 8, 10, 16]:
        run(df,k)
