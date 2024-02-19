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
from so3g.proj import Ranges, RangesMatrix

import soteralib
from sotodlib import core
from sotodlib.io.metadata import read_dataset

import h5py
from pixell import enmap, enplot


def restrict_highSN(aman, SNR_man):
    SNR_man.restrict('dets', SNR_man.dets.vals[SNR_man.SNR > 3.])
    aman.restrict('dets', SNR_man.dets.vals)
    return

def tod_process_v1(aman):
    print('detrend')
    tod_ops.detrend_tod(aman)
    
    print('flag turnaround')
    tod_ops.flags.get_turnaround_flags(aman, az=None, method='scanspeed', name='turnarounds',
                         merge=True, merge_lr=True, overwrite=True, 
                         t_buffer=2., kernel_size=400, peak_threshold=0.1, rel_distance_peaks=0.3,
                         truncate=False, qlim=1)
    
    print('compute source flag')
    coords.planets.compute_source_flags(aman, center_on='jupiter', max_pix=100000000,
                                   wrap='jupiter', mask={'shape':'circle', 'xyr':[0,0,1]})
    
    # Low pass filter
    print('lowpass filter')
    tod_ops.apodize_cosine(aman)
    filt = tod_ops.filters.get_lpf({'type': 'sine2', 'cutoff': 1.9, 'trans_width':0.1})
    aman.signal = tod_ops.fourier_filter(aman, filt, signal_name='signal')
    aman.restrict('samps', (aman.samps.offset + 2000, aman.samps.offset + aman.samps.count -2000))
    
    # glitch flagging
    print('glitch flagging')
    tod_ops.flags.get_glitch_flags(aman, overwrite=True)
    
    # remove jupiter signal from glitches
    aman.flags.wrap('anti_jupiter', ~aman.flags.jupiter)
    aman.flags.reduce(['anti_jupiter','glitches'], method='intersect', wrap=True, new_flag='glitches', remove_reduced=True)

    # remove too glithcy dets
    aman.restrict('dets', aman.dets.vals[np.mean(aman.flags.glitches.mask(), axis=1) < 0.01])
    
    
    # take union for mask for subscan_poly
    aman.flags.reduce(['jupiter', 'glitches'], method='union', wrap=True, new_flag='jupiter_and_glitches', remove_reduced=False)
    # Subscan polyfilter
    print('subscan polyfilter')
    degree = 10
    tod_ops.subscan_polyfilter(aman, degree, exclude_turnarounds=False, mask='jupiter_and_glitches', in_place=True)
    aman.restrict('samps', (aman.samps.offset, aman.samps.offset + aman.samps.count -200*30))
    
    return

def make_single_detmaps(aman, save_dir, file_name, cuts=None, res_deg=0.1):
    # plot out the single detmaps
    if cuts == None:
        cuts = RangesMatrix([ Ranges.from_bitmask(np.zeros(aman.samps.count, dtype=bool)) for di in range(aman.dets.count)])
    res = np.deg2rad(res_deg)
    P = coords.planets.get_scan_P(aman, planet='jupiter', res=res, cuts=cuts, threads=False)[0]
    for di, det in enumerate(tqdm(aman.dets.vals)):
        det_weights = np.zeros(aman.dets.count, dtype='float32')
        det_weights[di] = 1.
        mT_weighted = P.to_map(tod=aman, signal='signal', comps='T', det_weights=det_weights)
        wT = P.to_weights(aman, signal='signal', comps='T', det_weights=det_weights)
        mT = P.remove_weights(signal_map=mT_weighted, weights_map=wT, comps='T')[0]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        hdf_path = os.path.join(save_dir, file_name)
        enmap.write_hdf(hdf_path, mT, address=det)
    return

def main_SCR2(obs_id):
    telescope = 'satp1'
    slice_obs_files = slice(None, None)
    unit = 'pW'
    
    pointing_hdf = '/so/home/tterasaki/MF1/process/map2point/v4_240119_stable/combined_pointing_results/SCR2/all_ws.hdf'
    aman = soteralib.load_data.load_data_level2(obs_id, telescope, slice_obs_files=slice_obs_files, unit=unit,
                                                bgmap_style='last', iv_style='last', biasstep_style='last', 
                                                calc_PSD=True, load_acu=True, load_fake_focal_plane=False,
                                               load_pointing=True, pointing_hdf=pointing_hdf)
    
    # get SNR_man
    SNR_man = core.AxisManager.load(f'/so/home/tterasaki/summary/2402_abscal/MF1_SCR2/jupiter_SNR_SCR2/{obs_id}_SNRman.hdf')
    restrict_highSN(aman, SNR_man)
    
    # TOD processing
    tod_process_v1(aman)
    
    # Single detmap making
    cuts = aman.flags.glitches
    res_deg = 0.1
    save_dir = '/so/home/tterasaki/summary/2402_abscal/MF1_SCR2/single_detmaps_SCR2/'
    file_name = f'{obs_id}_single_detmaps.hdf'
    make_single_detmaps(aman, save_dir, file_name, cuts=cuts, res_deg=res_deg)
    return

        
    
