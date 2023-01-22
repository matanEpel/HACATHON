import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


def curve(x, a, b, c):
    return a + b * np.exp(-x * c)


def cantor_fit(distances, probs, plot=False):
    popt, pcov = curve_fit(curve, distances, probs, p0=(0.6, 0.6, 0.02))
    if plot:
        plt.scatter(distances, curve(distances, *popt), label=r'Fit: $%5.3f+ %5.3fexp^(-%5.3f+x)$' %
                                                              tuple(popt), zorder=0)
        plt.scatter(distances, probs, c='orange', label='Data', zorder=1)
        plt.legend()
        plt.xlabel(r'distance [# Bases]')
        plt.ylabel('probability')
        plt.show()
    return popt, pcov


def apply_fit(distances, probs, curve, plot=False):
    popt, pcov = cantor_fit(distances, probs, curve, plot)
    return curve(distances, *popt), r'Fit: $%5.3f+ %5.3fexp^(-%5.3f+x)$' % tuple(popt)


def return_curves(df):
    seq = []
    popt, pcov = cantor_fit(df["dist"].to_numpy(), df["0->0"].to_numpy(), True)
    seq.append(lambda x: curve(x, *popt))
    popt, pcov = cantor_fit(df["dist"].to_numpy(), df["0->1"].to_numpy(), True)
    seq.append(lambda x: curve(x, *popt))
    popt, pcov = cantor_fit(df["dist"].to_numpy(), df["1->0"].to_numpy(), True)
    seq.append(lambda x: curve(x, *popt))
    popt, pcov = cantor_fit(df["dist"].to_numpy(), df["1->1"].to_numpy(), True)
    seq.append(lambda x: curve(x, *popt))
    return seq


def apply_fit_on_df(df, plot=False):
    df['0->0 fit'],label00 = apply_fit(df["dist"].to_numpy(), df["0->1"].to_numpy(), plot)
    df['0->1 fit'], label01 = apply_fit(df["dist"].to_numpy(), df["0->0"].to_numpy(), plot)
    df['1->0 fit'],label10 = apply_fit(df["dist"].to_numpy(), df["1->0"].to_numpy(), plot)
    df['1->1 fit'],label11 = apply_fit(df["dist"].to_numpy(), df["1->1"].to_numpy(), plot)
    return df, label00, label01, label10, label11



