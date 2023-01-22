import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cantor_fit import return_curves
import plotly.express as px


def MLE_estimator_helper(d, row, curves):
    p00 = curves[0](d)
    p01 = curves[1](d)
    p10 = curves[2](d)
    p11 = curves[3](d)
    return (row["0->0"] * np.log2(p00) + row["0->1"] * np.log2(p01) + row["1->0"] * np.log2(p10) + row[
        "1->1"] * np.log2(
        p11))


def MLE_estimator(curves, df):
    eval_dist = []
    acordionicity = []
    for index, row in df.iterrows():
        print(index)
        d_ = np.linspace(1, 400, 401)
        new_d = d_[np.argmax(np.array([MLE_estimator_helper(d, row, curves) for d in d_]))]
        eval_dist.append(new_d)
        acordionicity.append(new_d / row["dist"])
    df["eval_dist"] = eval_dist
    df["acordionicity"] = acordionicity
    return df


def divide_by_prior(df):
    prior_0 = (df['0->0'] + df['0->1'])
    prior_1 = df['1->0'] + df['1->1']
    df['0->0'] /= prior_0
    df['0->1'] /= prior_0
    df['1->0'] /= prior_1
    df['1->1'] /= prior_1
    df = df.dropna()

    return df


def create_acordion_data(save=False, size=10000):
    locations = np.load("locations.npy")[:size]
    amount_in_pair = np.load("amount_in_pair.npy")[:size]

    dists = locations[1:] - locations[:-1]  # TODO: missing one line
    merged_data = np.hstack([dists.reshape(dists.shape[0], 1)[:-1], amount_in_pair[1:-1]])  # TODO: there are nulls
    df = pd.DataFrame(merged_data).dropna()
    df.columns = ['dist', '0->0', '0->1', '1->0', "1->1"]
    df = df[df["dist"] > 0].dropna()
    df = df[df["dist"] <= 400].dropna()  # TODO

    df_grouped = df.groupby("dist", as_index=False).sum()
    df_grouped = divide_by_prior(df_grouped)

    curves = return_curves(df_grouped)
    # df = divide_by_prior(df)
    df = MLE_estimator(curves, df)
    if save:
        df.to_pickle("acordion.pkl")
    return df


def get_acordion_data(create_new=True, size=10000, save=False, high_thresh=5, window_size=1000):
    if create_new:
        df = create_acordion_data(save=save, size=size)
    else:
        df = pd.read_pickle("acordion.pkl")

    new_df = df[np.abs(np.log(df["acordionicity"])) > np.log(high_thresh)]
    heatmap_data = []
    for i in range(len(df) - window_size):
        heatmap_data.append(len(new_df[(new_df.index.values > i) & (new_df.index.values < i + window_size)]))
    return df, heatmap_data


if __name__ == '__main__':
    df, heatmap = get_acordion_data(create_new=False, save=False)
    print(np.array(heatmap))

    a = np.array(heatmap)
    a= np.vstack([a for _ in range(300)])
    fig = px.imshow(img=a)
    fig.layout.width = 1500

    fig.show()
