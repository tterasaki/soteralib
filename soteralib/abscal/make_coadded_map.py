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

def restrict_highSN(aman, SNR_man):
    SNR_man.restrict('dets', SNR_man.dets.vals[SNR_man.SNR > 2.])
    aman.restrict('dets', SNR_man.dets)
    return

def main_SCR2(aman, ):
    telescope = 'satp1'
    slice_obs_files = slice(None, None)
    unit = 'pA'
    aman = soteralib.load_data.load_data_level2(obs_id, telescope, slice_obs_files=slice_obs_files, unit=unit,
                        bgmap_style='last', iv_style='last', biasstep_style='last', 
                        calc_PSD=True, load_acu=True, load_fake_focal_plane=True)
    pointing_hdf = '/so/home/tterasaki/MF1/process/map2point/v4_240119_stable/combined_pointing_results/SCR2/all_ws.hdf'
    wrap_pointing(aman, pointing_hdf)
    
    