import numpy as np
import pandas as pd
import plotly.express as px


def plot_all():
    locations = np.load("locations.npy")
    amount_in_pair = np.load("amount_in_pair.npy")

    dists = locations[1:] - locations[:-1]  # TODO: missing one line
    merged_data = np.hstack([dists.reshape(dists.shape[0], 1)[:-1], amount_in_pair[1:-1]])  # TODO: there are nulls
    df = pd.DataFrame(merged_data).dropna()
    df.columns = ['dist', '0->0', '0->1', '1->0', "1->1"]
    df = df.groupby("dist", as_index=False).sum()
    df = df[df["dist"] > 0].dropna()
    prior_0 = (df['0->0'] + df['0->1'])
    prior_1 = df['1->0'] + df['1->1']
    df['0->0'] /= prior_0
    df['0->1'] /= prior_0
    df['1->0'] /= prior_1
    df['1->1'] /= prior_1
    df = df.dropna()
    print(df)

    fig = px.scatter(df, x='dist', y=df.columns)
    fig.show()


if __name__ == '__main__':
    plot_all()
