import cmath
import numpy as np

# Reference: https://qiita.com/kn1cht/items/6be4f9b7ff2da940ca68
def circular_mean(angles, deg=False, axis=None):
    """
    Calculate the circular mean of angle data along a specified axis.

    Parameters:
    angles (array_like): Array of angles.
    deg (bool, optional): If True, the input angles are in degrees, otherwise in radians. Default is False.
    axis (int, optional): Axis along which the means are computed. Default is None, which computes the mean of the flattened array.

    Returns:
    float or ndarray: Circular mean of the input angles.
    """
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.exp(a * 1j)
    
    def _mean(angles_complex):
        mean = cmath.phase(angles_complex.sum()) # -pi < mean < pi
        return mean
    
    if axis is None:
        mean = _mean(angles_complex)
    else:
        mean = np.apply_along_axis(lambda x: _mean(x), axis, angles_complex)
    
    return np.rad2deg(mean) if deg else mean

def circular_std(angles, deg=False, axis=None):
    """
    Calculate the circular standard deviation of angle data along a specified axis.

    Parameters:
    angles (array_like): Array of angles.
    deg (bool, optional): If True, the input angles are in degrees, otherwise in radians. Default is False.
    axis (int, optional): Axis along which the standard deviations are computed. Default is None, which computes the standard deviation of the flattened array.

    Returns:
    float or ndarray: Circular standard deviation of the input angles.
    """
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.exp(a * 1j)
    
    def _std(angles_complex):
        r = abs(angles_complex.sum()) / len(angles_complex)
        std = np.sqrt(-2 * np.log(r))
        return std
    
    if axis is None:
        std = _std(angles_complex)
    else:
        std = np.apply_along_axis(lambda x: _std(x), axis, angles_complex)
    
    return np.rad2deg(std) if deg else std
