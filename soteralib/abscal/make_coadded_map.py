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

def make_coadd_detmap(aman, save_dir, file_name, cuts=None, res_deg=0.1):
    # plot out the single detmaps
    if cuts == None:
        cuts = RangesMatrix([ Ranges.from_bitmask(np.zeros(aman.samps.count, dtype=bool)) for di in range(aman.dets.count)])
    res = np.deg2rad(res_deg)
    P = coords.planets.get_scan_P(aman, planet='jupiter', res=res, cuts=cuts, threads=False)[0]
    
        
    det_weights = None
    mT_weighted = P.to_map(tod=aman, signal='signal', comps='T', det_weights=det_weights)
    wT = P.to_weights(aman, signal='signal', comps='T', det_weights=det_weights)
    mT = P.remove_weights(signal_map=mT_weighted, weights_map=wT, comps='T')[0]
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hdf_path = os.path.join(save_dir, file_name)
    enmap.write_hdf(hdf_path, mT)
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
    SNR_man = core.AxisManager.load(f'/so/home/tterasaki/summary/2402_abscal/MF1_SCR2/beaminfo_SCR2_wmapSNR/{obs_id}.hdf')
    aman.restrict('dets', SNR_man.dets.vals)
    aman.wrap('SNR', SNR_man.SNR, [(0, 'dets')])
    aman.restrict('dets', aman.dets.vals[aman.SNR > 3.], [(0, 'dets')])
    aman.restrict('dets', aman.dets.vals[~np.isnan(aman.biasstep.si)])
    aman.restrict('dets', aman.dets.vals[(-3e7 < aman.biasstep.si) & 
                                         (aman.biasstep.si < -0.1e7)])
    
    
    # detector selection
    aman.restrict('dets', aman.dets.vals[(np.nanpercentile(aman.wn, 5) < aman.wn) & \
                                         (aman.wn < np.nanpercentile(aman.wn, 95))])
    
    # TOD processing
    tod_process_v1(aman, apply_fourier_filt=False, remove_glitchy=True, glitchy_th=0.01)
    
    # map making
    res_deg = 0.1
    save_dir = '/so/home/tterasaki/summary/2402_abscal/MF1_SCR2/coadded_map_SCR2/'
    
    # 90
    aman_90 = aman.restrict('dets', aman.dets.vals[aman.is90], in_place=False)
    print(f'{obs_id}, 90GHz dets, #={aman_90.dets.count}')
    #file_name = f'{obs_id}_coadd_map_90.hdf'
    #cuts = aman_90.flags.glitches
    #make_coadd_detmap(aman_90, save_dir, file_name, cuts=cuts, res_deg=res_deg)
    
    # 150
    aman_150 = aman.restrict('dets', aman.dets.vals[aman.is150], in_place=False)
    print(f'{obs_id}, 150GHz dets, #={aman_150.dets.count}')
    #file_name = f'{obs_id}_coadd_map_150.hdf'
    #cuts = aman_150.flags.glitches
    #make_coadd_detmap(aman_150, save_dir, file_name, cuts=cuts, res_deg=res_deg)
    
    return


