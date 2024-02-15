import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sotodlib
from sotodlib import tod_ops, coords
from sotodlib.io import load_smurf as ls
from sotodlib.io.load_smurf import Observations, Files, TuneSets, Tunes
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn
from sotodlib.io import g3tsmurf_utils, hk_utils
from sodetlib.operations.iv import IVAnalysis
from sotodlib import coords

import soteralib
from sotodlib import core
from sotodlib.io.metadata import read_dataset

import h5py
from pixell import enmap, enplot

def wrap_pointing(aman, pointing_hdf):
    rset = read_dataset(pointing_hdf, 'focalplane')
    for di, det in enumerate(aman.dets.vals):
        map_idx = np.where(rset['dets:readout_id']==det)[0][0]
        xi = rset['xi'][map_idx]
        eta = rset['eta'][map_idx]
        aman.focal_plane.xi[di] = xi
        aman.focal_plane.eta[di] = eta
    aman.restrict('dets', aman.dets.vals[~np.isnan(aman.focal_plane.xi)])
    return
    
def tod_process_v1(aman):
    tod_ops.detrend_tod(aman)
    tod_ops.apodize_cosine(aman)
    filt = tod_ops.filters.get_bpf({'type': 'sine2', 'center': 1.0,
                                    'width': 1.8, 'trans_width':0.1})
    aman.signal = tod_ops.fourier_filter(aman, filt, signal_name='signal')
    
    aman.restrict('samps', (2000, -2000))
    coords.planets.compute_source_flags(aman, center_on='jupiter', max_pix=100000000,
                                   wrap='jupiter', mask={'shape':'circle', 'xyr':[0,0,1]})
    return

def get_SNR(aman):
    S_array = np.nanpercentile(np.where(~aman.flags.jupiter.mask(), np.nan, aman.signal), 99.9, axis=-1)
    N_array = np.nanstd(np.where(aman.flags.jupiter.mask(), np.nan, aman.signal), axis=-1)
    SNR_array = S_array/N_array
    return SNR_array

def main_SCR2(obs_id):
    telescope = 'satp1'
    slice_obs_files = slice(None, None)
    unit = 'pA'
    aman = soteralib.load_data.load_data_level2(obs_id, telescope, slice_obs_files=slice_obs_files, unit=unit,
                        bgmap_style='last', iv_style='last', biasstep_style='last', 
                        calc_PSD=True, load_acu=True, load_fake_focal_plane=True)
    pointing_hdf = '/so/home/tterasaki/MF1/process/map2point/v4_240119_stable/combined_pointing_results/SCR2/all_ws.hdf'
    wrap_pointing(aman, pointing_hdf)
    
    tod_process_v1(aman)
    SNR_array = get_SNR(aman)
    
    SNR_man = core.AxisManager(aman.dets)
    SNR_man.wrap('SNR', SNR_array, [(0, 'dets')])
    
    save_dir = 'jupiter_SNR_SCR2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    SNRman = core.AxisManager(aman.dets)
    SNRman.wrap('SNR', SNR_array, [(0, 'dets')])
    
    save_file = os.path.join(save_dir, f'{obs_id}_SNRman.hdf')
    SNR_man.save(save_file)
    print(f'Saved: {save_file}')
    return

def main_Run11(obs_id):
    telescope = 'satp3'
    slice_obs_files = slice(None, None)
    unit = 'pA'
    aman = soteralib.load_data.load_data_level2(obs_id, telescope, slice_obs_files=slice_obs_files, unit=unit,
                        bgmap_style='last', iv_style='last', biasstep_style='last',
                        calc_PSD=True, load_acu=True, load_fake_focal_plane=True)

    pointing_hdf = '/so/home/tterasaki/MF2/process/map2point/v4_240119_stable/combined_pointing_results/run11/all_ws.hdf'
    wrap_pointing(aman, pointing_hdf)
    
    tod_process_v1(aman)
    SNR_array = get_SNR(aman)
    
    SNR_man = core.AxisManager(aman.dets)
    SNR_man.wrap('SNR', SNR_array, [(0, 'dets')])
    
    save_dir = 'jupiter_SNR_Run11'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    SNRman = core.AxisManager(aman.dets)
    SNRman.wrap('SNR', SNR_array, [(0, 'dets')])
    
    save_file = os.path.join(save_dir, f'{obs_id}_SNRman.hdf')
    SNR_man.save(save_file)
    print(f'Saved: {save_file}')
    
    return
