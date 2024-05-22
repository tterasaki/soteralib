import os
import glob
import h5py
import numpy as np
import soteralib as tera
from scipy.constants import k

def get_detector_bandpass(band_name):
    """
    Example
    -------
    import soteralib as tera
    freqs, trans = tera.utils.bandpass.get_detector_bandpass('f090')
    """
    assert band_name in ['f090', 'f150']
    bandpass_dir = os.path.join(os.path.dirname(tera.__file__), 'data', 'detector_bandpass')
    hdf_file = glob.glob(os.path.join(bandpass_dir, f'{band_name}*'))[0]
    with h5py.File(hdf_file) as f:
        freqs = f['freqs'][...]
        trans = f['transmission'][...]
    return freqs, trans

def band_average(nu_spectrum, spectrum, nu_transmission, transmission):
    """
    Calcurate band average of spectrum for a given transmission.
    Parameters:
        nu_spectrum: frequency array of spectrum [Hz]
        spectrum: spectrum [X]
        nu_transmission: frequency array of transmission [Hz]
        transmission: transmission with no unit
    """
    # interpolate transmission to nu_spectrum
    transmission = np.interp(nu_spectrum, nu_transmission, transmission)
    # calculate band average
    band_average = np.trapz(transmission * spectrum, nu_spectrum) / np.trapz(transmission, nu_spectrum)
    return band_average

def get_bandwidth(band_name):
    """
    f090: 27.09e9 [Hz]
    f150: 36.40e9 [Hz]
    """
    freqs, trans = get_detector_bandpass(bandname)
    trans /= trans.max()
    return np.trapz(trans, freqs)

def get_pW_per_Krj(band_name, eta=1.):
    """
    Assuming eta=1 if not specified.
    """
    bandwidth = get_bandwidth(band_name)
    pW_per_Krj = eta * k * 1e12 * bandwidth
    return pW_per_Krj

def get_band_averaged_Trj2Tkcmb(band_name):
    """
    f090: 1.248
    f150: 1.714
    """
    freqs, trans = tera.utils.bandpass.get_detector_bandpass(bandname)
    return tera.utils.bandpass.band_average(freqs, tera.utils.Trj2Tkcmb(1, freqs), freqs, trans)

def get_pW_per_Kcmb(band_name, eta=1.):
    pW_per_Krj = get_pW_per_Krj(band_name=band_name, eta=eta)
    Kcmb_per_Krj = get_band_averaged_Trj2Tkcmb(band_name)
    pW_per_Kcmb = pW_per_Krj / Kcmb_per_Krj
    return pW_per_Kcmb
