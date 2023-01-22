import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cantor_fit import return_curves


def MLE_estimator_helper(d, row, curves):
    p00 = curves[0](d)
    p01 = curves[1](d)
    p10 = curves[2](d)
    p11 = curves[3](d)
    return row["0->0"] * np.log2(p00) + row["0->1"] * np.log2(p01) + row["1->0"] * np.log2(p10) + row["1->1"] * np.log2(
        p11)


def MLE_estimator(curves, df):
    # plt.plot(np.arange(400), curves[0](np.arange(400)))
    # plt.plot(np.arange(400), curves[1](np.arange(400)))
    # plt.plot(np.arange(400), curves[2](np.arange(400)))
    # plt.plot(np.arange(400), curves[3](np.arange(400)))
    #
    # plt.show()
    eval_dist = []
    for index, row in df.iterrows():
        d_ = np.linspace(0, max(row["dist"] * 3, row["dist"] + 15), 40)
        eval_dist.append(d_[np.argmax(np.array([MLE_estimator_helper(d, row, curves) for d in d_]))])
    df["eval_dist"] = eval_dist
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


if __name__ == '__main__':
    locations = np.load("locations.npy")[:10000]
    amount_in_pair = np.load("amount_in_pair.npy")[:10000]

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
    print(df)
