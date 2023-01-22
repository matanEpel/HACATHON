import numpy as np
import pandas as pd


def plot_all():
    locations = np.load("locations.npy")
    amount_in_pair = np.load("amount_in_pair.npy")
    amount_in_pair = amount_in_pair / amount_in_pair.sum(axis=1)[:, None]
    dists = locations[1:] - locations[:-1]  # TODO: missing one line
    merged_data = np.hstack([dists.reshape(dists.shape[0], 1), amount_in_pair[1:]])

    df = pd.DataFrame(merged_data)
    df.columns = ['dist', '0->0', '0->1', '1->1', "1->0"]
    df = df.groupby("dist").sum()
    print(df)




if __name__ == '__main__':
    plot_all()
