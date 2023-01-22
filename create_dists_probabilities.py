import numpy as np
import pandas as pd
import plotly.express as px
from cantor_fit import apply_fit_on_df, apply_fit_on_df_matrix_rate


def plot_all():
    locations = np.load("locations.npy")
    # amount_in_pair = np.load("amount_in_pair.npy")
    amount_in_pair = np.load("amount_in_pair.npy")
    dists = locations[1:] - locations[:-1]  # TODO: missing one line
    merged_data = np.hstack([dists.reshape(dists.shape[0], 1)[:-1], amount_in_pair[1:-1]])  # TODO: there are nulls
    # merged_data = np.hstack([dists.reshape(dists.shape[0], 1)[:-1], amount_in_pair[1:]])  # TODO: there are nulls

    df = pd.DataFrame(merged_data).dropna()
    df.columns = ['dist', '0->0', '0->1', '1->0', "1->1"]
    df = df[df["dist"] > 0].dropna()
    df = df[df["dist"] <= 400].dropna()
    # for f_name in ["data_inside_inside.pkl","data_outside_inside.pkl","data_inside_outside.pkl","data_outside_outside.pkl"]:
    #     df = pd.read_pickle(f_name)
    # df.to_pickle("./final_samples.pkl")

    df = df.groupby("dist", as_index=False).sum()
    prior_0 = (df['0->0'] + df['0->1'])
    prior_1 = df['1->0'] + df['1->1']
    df['0->0'] /= prior_0
    df['0->1'] /= prior_0
    df['1->0'] /= prior_1
    df['1->1'] /= prior_1
    df = df.dropna()
    df = apply_fit_on_df_matrix_rate(df, False)
    print(df.columns)
    # df1 = apply_fit_on_df(df, False)
    fig = px.scatter(df, x='dist', y=df.columns)
    fig.show()


if __name__ == '__main__':
    plot_all()
