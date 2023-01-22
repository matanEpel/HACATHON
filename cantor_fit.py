import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def curve(x, a, b, c):
    return a + b * np.exp(-c * x)


def cantor_fit(distances, probs, plot = False):
    popt, pcov = curve_fit(curve, distances, probs)
    if plot:
        plt.scatter(distances, curve(distances, *popt), label=r'Fit: $%5.3f+ %5.3fexp^(-%5.3f+x)$' %
                                                              tuple(popt), zorder=0)
        plt.scatter(distances, probs, c='orange', label='Data', zorder=1)
        plt.legend()
        plt.xlabel(r'distance [# Bases]')
        plt.ylabel('probability')
        plt.show()
    return popt, pcov