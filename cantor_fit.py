import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


def curve0(x, a, b, c):
    return a - b * np.exp(-x * c)


def curve1(x, a, b, c):
    return a + b * np.exp(-x * c)


def curve_rate_matrix(flag):  # TODO
    if flag == "00":
        return lambda t, a, b: np.exp(-t * (a + b)) * a / (a + b) + b / (a + b)
    elif flag == "01":
        return lambda t, a, b: -np.exp(-t * (a + b)) * a / (a + b) + a / (a + b)
    elif flag == "10":
        return lambda t, a, b: -np.exp(-t * (a + b)) * b / (a + b) + b / (a + b)
    elif flag == "11":
        return lambda t, a, b: np.exp(-t * (a + b)) * b / (a + b) + a / (a + b)
    return


def cantor_fit(distances, probs, curve, plot=False, p0=(0.6, 0.6, 0.02)):
    popt, pcov = curve_fit(curve, distances, probs, p0=p0, bounds=(-1, 1), maxfev=5000)
    if plot:
        plt.scatter(distances, curve(distances, *popt), label=r'Fit: $%5.3f+ %5.3fexp^{-%5.3fx}$' %
                                                              tuple(popt), zorder=0)
        plt.scatter(distances, probs, c='orange', label='Data', zorder=1)
        plt.legend()
        plt.xlabel(r'distance [# Bases]')
        plt.ylabel('probability')
        plt.show()
    return popt, pcov


def apply_fit(distances, probs, curve, p0, plot=False):
    popt, pcov = cantor_fit(distances, probs, curve, plot, p0)
    return curve(distances, *popt), r'Fit: $%5.3f+ %5.3fexp^{-%5.3fx}$' % tuple(popt)

def apply_fit_rate_matrix(distances, probs, curve, p0, plot=False):
    popt, pcov = cantor_fit(distances, probs, curve, plot, p0)
    return curve(distances, *popt), r'Fit: $\alpha=%5.3f, \beta=%5.3f$' % tuple(popt)


def return_curves(df):
    seq = []
    popt, pcov = cantor_fit(df["dist"].to_numpy(), df["0->0"].to_numpy(), curve0, False)
    seq.append(lambda x: curve0(x, *popt))
    popt1, pcov = cantor_fit(df["dist"].to_numpy(), df["0->1"].to_numpy(), curve1, False)
    seq.append(lambda x: curve1(x, *popt1))
    popt2, pcov = cantor_fit(df["dist"].to_numpy(), df["1->0"].to_numpy(), curve0, False)
    seq.append(lambda x: curve0(x, *popt2))
    popt3, pcov = cantor_fit(df["dist"].to_numpy(), df["1->1"].to_numpy(), curve1, False)
    seq.append(lambda x: curve1(x, *popt3))
    return seq


def apply_fit_on_df(df, plot=False):
    df['0->0 fit'], label00 = apply_fit(df["dist"].to_numpy(), df["0->0"].to_numpy(), curve0, p0=(0.6, 0.6, 0.02),
                                        plot=plot)
    df['0->1 fit'], label01 = apply_fit(df["dist"].to_numpy(), df["0->1"].to_numpy(), curve1, p0=(0.6, 0.6, 0.02),
                                        plot=plot)
    df['1->0 fit'], label10 = apply_fit(df["dist"].to_numpy(), df["1->0"].to_numpy(), curve0, p0=(0.6, 0.6, 0.02),
                                        plot=plot)
    df['1->1 fit'], label11 = apply_fit(df["dist"].to_numpy(), df["1->1"].to_numpy(), curve1, p0=(0.6, 0.6, 0.02),
                                        plot=plot)
    df.rename(columns={'0->0 fit': label00, '0->1 fit': label01, '1->0 fit': label10, '1->1 fit': label11},
              inplace=True)

    return df


def apply_fit_on_df_matrix_rate(df, plot=False):
    df['0->0 fit'], label00 = apply_fit_rate_matrix(df["dist"].to_numpy(), df["0->0"].to_numpy(), curve_rate_matrix("00"),
                                        p0=(0.002, 0.008), plot=plot)
    df['0->1 fit'], label01 = apply_fit_rate_matrix(df["dist"].to_numpy(), df["0->1"].to_numpy(), curve_rate_matrix("01"),
                                        p0=(0.002, 0.008), plot=plot)
    df['1->0 fit'], label10 = apply_fit_rate_matrix(df["dist"].to_numpy(), df["1->0"].to_numpy(), curve_rate_matrix("10"),
                                        p0=(0.002, 0.008), plot=plot)
    df['1->1 fit'], label11 = apply_fit_rate_matrix(df["dist"].to_numpy(), df["1->1"].to_numpy(), curve_rate_matrix("11"),
                                        p0=(0.002, 0.008), plot=plot)
    df.rename(columns={'0->0 fit': '0->0 fit:'+label00, '0->1 fit': '0->1 fit:'+label01, '1->0 fit': '1->0 fit:'+label10, '1->1 fit': '1->1 fit:'+label11},
              inplace=True)

    return df


if __name__ == '__main__':
    d = np.linspace(1, 400, 400)
    # plt.plot(d, curve_rate_matrix("00")(d, 0.002, 0.008), label="00")
    # plt.plot(d, curve_rate_matrix("01")(d, 0.2, 0.8), label="01")
    # plt.plot(d, curve_rate_matrix("10")(d, 0.2, 0.8), label="10")
    # plt.plot(d, curve_rate_matrix("11")(d, 0.2, 0.8), label="11")
    # plt.legend()
    # plt.show()
