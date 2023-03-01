import cmath
import numpy as np
import scipy.stats as stats

# Reference: https://qiita.com/kn1cht/items/6be4f9b7ff2da940ca68
def circular_mean(angles, deg=False):
    """
    Circular mean of angle data
    """
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) # -pi < mean < pi
    return np.rad2deg(mean) if deg else mean

def circular_std(angles, deg=False):
    """
    Circular standard deviation of angle data
    """
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r = abs(angles_complex.sum()) / len(angles)
    std = np.sqrt(-2 * np.log(r))
    return np.rad2deg(std) if deg else std