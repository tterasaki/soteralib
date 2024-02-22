import numpy as np
from pixell import utils

fwhm2sigma = 1/2.3548

def sigma2omega(sigma):
    omega = 4 * np.pi * (1 - np.exp(-0.5 * sigma**2))
    return omega

