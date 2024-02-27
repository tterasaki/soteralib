import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soteralib as tera

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

def tod_process_v1(aman, apply_fourier_filt=False, remove_glitchy=False, glitchy_th=0.01):
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
    
    # Low pass filter if specified
    tod_ops.apodize_cosine(aman)
    if apply_fourier_filt:
        print('lowpass filter')
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
    if remove_glitchy:
        aman.restrict('dets', aman.dets.vals[np.mean(aman.flags.glitches.mask(), axis=1) < glitchy_th])
    
    # take union for mask for subscan_poly
    aman.flags.reduce(['jupiter', 'glitches'], method='union', wrap=True, new_flag='jupiter_and_glitches', remove_reduced=False)
    # Subscan polyfilter
    print('subscan polyfilter')
    degree = 10
    tod_ops.subscan_polyfilter(aman, degree, exclude_turnarounds=False, mask='jupiter_and_glitches', in_place=True)
    aman.restrict('samps', (aman.samps.offset + 200 * 60, aman.samps.offset + aman.samps.count - 200*60))
    
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
    unit = 'pA'
    
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

def main_Run11(obs_id):
    telescope = 'satp3'
    slice_obs_files = slice(None, None)
    unit = 'pA'
    
    pointing_hdf = '/so/home/tterasaki/MF2/process/map2point/v4_240119_stable/combined_pointing_results/run11/all_ws.hdf'
    aman = soteralib.load_data.load_data_level2(obs_id, telescope, slice_obs_files=slice_obs_files, unit=unit,
                                                bgmap_style='last', iv_style='last', biasstep_style='last', 
                                                calc_PSD=True, load_acu=True, load_fake_focal_plane=False,
                                               load_pointing=True, pointing_hdf=pointing_hdf)
    
    # get SNR_man
    SNR_man = core.AxisManager.load(f'/so/home/tterasaki/summary/2402_abscal/MF2_Run11/jupiter_SNR_Run11/{obs_id}_SNRman.hdf')
    restrict_highSN(aman, SNR_man)
    
    # TOD processing
    tod_process_v1(aman)
    
    # Single detmap making
    cuts = aman.flags.glitches
    res_deg = 0.1
    save_dir = '/so/home/tterasaki/summary/2402_abscal/MF2_Run11/single_detmaps_Run11/'
    file_name = f'{obs_id}_single_detmaps.hdf'
    make_single_detmaps(aman, save_dir, file_name, cuts=cuts, res_deg=res_deg)
    return

def wrap_isband_level3(aman):
    det_types = []
    for _did in aman.det_info.det_id:
        _det_type = _did.split('_')[1]
        if _det_type == 'MATCH':
            det_types.append('no_match')
        elif _det_type == 'DARK':
            det_types.append('dark')
        else:
            det_types.append(_det_type)

    det_types = np.array(det_types)
    aman.wrap('det_types', det_types, [(0, 'dets')])

    is90_match = aman.det_types=='f090'
    is150_match = aman.det_types=='f150'
    aman.wrap('is90', is90_match, [(0, 'dets')])
    aman.wrap('is150', is150_match, [(0, 'dets')])
    return
    
def main_SCR2_level3(obs_id, ws):
    #obs_id = 'obs_1698990364_satp1_1111111'
    
    ctx_file = '/so/home/tterasaki/summary/2402_abscal/MF1_SCR2_level3/context.yaml'
    ctx = core.Context(ctx_file)
    meta = ctx.get_meta(obs_id)
    meta.restrict('dets', meta.dets.vals[meta.det_info.wafer_slot==ws])
    aman = ctx.get_obs(meta)
    aman.restrict('dets', aman.dets.vals[~np.isnan(aman.focal_plane.xi)])
    aman.signal *= tera.load_data.phase_to_pA
    
    # wrap is90 and is150
    wrap_isband_level3(aman)
    
    # TOD processing
    tod_process_v1(aman)
    
    # Single detmap making
    cuts = aman.flags.glitches
    res_deg = 0.1
    save_dir = '/so/home/tterasaki/summary/2402_abscal/MF1_SCR2_level3/single_detmaps_SCR2/'
    file_name = f'{obs_id}_{ws}_single_detmaps.hdf'
    make_single_detmaps(aman, save_dir, file_name, cuts=cuts, res_deg=res_deg)
    return