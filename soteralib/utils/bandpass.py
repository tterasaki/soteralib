import os
import glob
import h5py
import numpy as np
import soteralib as tera

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