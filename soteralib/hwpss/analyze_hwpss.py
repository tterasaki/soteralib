import os
import sys
import glob
from tqdm import tqdm
import argparse
import numpy as np
import scipy
import argparse

from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew
from lmfit import Model
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from sotodlib import coords, core, tod_ops, hwp
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn
from so3g.proj import Ranges, RangesMatrix
from scipy.constants import k
cm = matplotlib.colormaps.get_cmap('viridis')
from sodetlib.operations.iv import IVAnalysis
from sotodlib.io.load_book import load_smurf_npy_data, get_cal_obsids
from sotodlib.hwp.hwp_angle_model import apply_hwp_angle_model
from sotodlib.tod_ops import t2pleakage

nperseg = 200*1000
bandwidth_f090 = 27.1 * 1e9
bandwidth_f150 = 36.4 * 1e9
pW_per_Krj_f090_full = k*bandwidth_f090 * 1e12
pW_per_Krj_f150_full = k*bandwidth_f150 * 1e12
pwv_ref = np.array([0.          , 0.3333333333, 0.6666666667, 1.          , 1.3333333333,
                    1.6666666667, 2.          , 2.3333333333, 2.6666666667, 3.          ])
Tatm_rj_f090_ref = np.array([ 6.3556491566,  6.7564397072,  7.1697486877,  7.5955414909,  8.033768948 , 
                             8.4843738835,  8.9472935975,  9.422517075 ,  9.9102161784, 10.410106652 ])
Tatm_rj_f150_ref = np.array([ 3.3205340822,  4.6600697112,  6.0202319949,  7.4021476475,  8.8062527532,
                             10.2326073651, 11.6810593533, 13.1517261051, 14.6459678275, 16.1618460923])
p_zenith_100eff_slope_f090 = np.polyfit(pwv_ref, Tatm_rj_f090_ref * pW_per_Krj_f090_full, deg=1)[0]
p_zenith_100eff_slope_f150 = np.polyfit(pwv_ref, Tatm_rj_f150_ref * pW_per_Krj_f150_full, deg=1)[0]


def load_data(ctx, obs_id, ws, bandpass, debug=False, 
              do_correct_iir=True,
              do_rn_selection=True, 
              do_calibration=True, 
              do_correct_hwp=True):
    meta = ctx.get_meta(obs_id, dets={'wafer_slot':ws, 'wafer.bandpass': bandpass})
    
    # data cuts
    valid_fp = np.sum(np.isnan([meta.focal_plane.xi, 
                                meta.focal_plane.eta, 
                                meta.focal_plane.gamma]).astype(int), axis=0) == 0
    meta.restrict('dets', meta.dets.vals[valid_fp])
    
    if do_rn_selection:
        if 'det_cal' in meta._fields.keys():
            meta.restrict('dets', meta.dets.vals[(0.2<meta.det_cal.r_frac)&(meta.det_cal.r_frac<0.8)])
        else:
            raise ValueError
    if debug:
        meta.restrict('dets', meta.dets.vals[:debug])
    
    aman = ctx.get_obs(meta)
    #_ = tod_ops.flags.get_turnaround_flags(aman, t_buffer=2.0, truncate=True)
    
    aman.focal_plane.gamma = np.arctan(np.tan(aman.focal_plane.gamma))
    if do_calibration:
        if 'det_cal' in aman._fields.keys():
            aman.restrict('dets', aman.dets.vals[aman.det_cal.phase_to_pW > 0])
            aman.signal *= aman.det_cal.phase_to_pW[:, np.newaxis]
        else:
            raise ValueError
    
    if do_correct_iir:
        iir_filt = tod_ops.filters.iir_filter(invert=True)
        aman.signal = tod_ops.fourier_filter(aman, iir_filt)
        aman.restrict('samps', (aman.samps.offset+1000, aman.samps.offset+aman.samps.count-1000))
        
    
    if do_correct_hwp:
        apply_hwp_angle_model(aman)    
    
    return aman


# Data cuts
def calc_wrap_psd(aman, signal_name, merge=False, merge_wn=False, merge_suffix=None, low_f=2.0, high_f=5.0):
    freqs, Pxx = calc_psd(aman, signal=aman[signal_name], nperseg=nperseg, merge=False)
    if merge:
        assert merge_suffix is not None
        if 'nusamps' not in list(aman._axes.keys()):
            nusamps = core.OffsetAxis('nusamps', len(freqs))
            aman.wrap_new('freqs', (nusamps, ))
            aman.freqs = freqs
        aman.wrap(f'Pxx_{merge_suffix}', Pxx, [(0, 'dets'), (1, 'nusamps')])
    if merge_wn:
        wn = calc_wn(aman, pxx=Pxx, freqs=freqs, low_f=low_f, high_f=high_f)
        _ = aman.wrap(f'wn_{merge_suffix}', wn, [(0, 'dets')])
    return  freqs, Pxx

def ptp_cuts(aman, signal_name, kurtosis_threshold=2., print_state=False):
    print(f'dets before ptp cuts: {aman.dets.count}')
    ptps = np.ptp(aman[signal_name], axis=1)
    ratio = ptps/np.median(ptps)
    outlier_mask = (ratio<0.5) | (2.<ratio)
    aman.restrict('dets', aman.dets.vals[~outlier_mask])
    
    print(f'dets after rough ptp cuts: {aman.dets.count}')
    ptps = np.ptp(aman[signal_name], axis=1)
    kurtosis_ptp = kurtosis(ptps)    
    while True:
        ptps = np.ptp(aman[signal_name], axis=1)
        kurtosis_ptp = kurtosis(ptps)
        if np.abs(kurtosis_ptp) < kurtosis_threshold:
            print(f'dets after ptp cuts:{aman.dets.count}, ptp_kurt: {kurtosis_ptp:.1f}')
            break
        else:
            max_is_bad_factor = np.max(ptps)/np.median(ptps)
            min_is_bad_factor = np.median(ptps)/np.min(ptps)
            if max_is_bad_factor > min_is_bad_factor:
                aman.restrict('dets', aman.dets.vals[ptps < np.max(ptps)])
            else:
                aman.restrict('dets', aman.dets.vals[ptps > np.min(ptps)])
            if print_state:
                print(f'dets:{aman.dets.count}, ptp_kurt: {kurtosis_ptp:.1f}')
    return
    
def get_glitches(aman):
    glitches_T = tod_ops.flags.get_glitch_flags(aman, signal_name='dsT', merge=True, name='glitches_T')
    glitches_Q = tod_ops.flags.get_glitch_flags(aman, signal_name='demodQ', merge=True, name='glitches_Q')
    glitches_U = tod_ops.flags.get_glitch_flags(aman, signal_name='demodU', merge=True, name='glitches_U')
    aman.flags.reduce(flags=['glitches_T', 'glitches_Q', 'glitches_U'], method='union', wrap=True, new_flag='glitches', remove_reduced=True)
    return 

# TOD process
def tod_process(aman, subtract_hwpss=False):
    # get hwpss
    print('getting hwpss')
    _ = hwp.get_hwpss(aman)
    
    print('detrend')
    tod_ops.detrend_tod(aman, method='median')
    
    if subtract_hwpss:
        print('subtract hwpss')
        hwp.subtract_hwpss(aman)
        aman.signal = aman.hwpss_remove
        aman.move('hwpss_remove', None)
        
    tod_ops.apodize_cosine(aman, apodize_samps=10000)
    calc_wrap_psd(aman, signal_name='signal', merge=True, merge_wn=True, merge_suffix='signal',
                  low_f=2.0, high_f=5.0,)
    
    print('demod')
    hwp.demod_tod(aman)
    calc_wrap_psd(aman, signal_name='dsT', merge=True, merge_wn=True, merge_suffix='dsT', low_f=1.0, high_f=1.5)
    calc_wrap_psd(aman, signal_name='demodQ', merge=True, merge_wn=True, merge_suffix='demodQ', low_f=1.0, high_f=1.5)
    calc_wrap_psd(aman, signal_name='demodU', merge=True, merge_wn=True, merge_suffix='demodU', low_f=1.0, high_f=1.5)
    aman.restrict('samps', (aman.samps.offset+10000, aman.samps.offset + aman.samps.count-10000))
    
    print('ptp cuts')
    ptp_cuts(aman, signal_name='dsT', kurtosis_threshold=2., print_state=False)
    aman.wrap('pwv_median', np.nanmedian(aman.pwv_class))
    fhwp = -(np.sum(np.diff(np.unwrap(aman.hwp_angle))) /(aman.timestamps[-1] - aman.timestamps[0])) / (2 * np.pi)
    aman.wrap('fhwp', fhwp)
    
    print('glitch')
    get_glitches(aman)
    return aman


# POLARIZATION
def rotate_QU(aman, Q_name, U_name, sign_rot=1):
    """
    rotate the Q and U values with detector angles. The sign_rot specifies the direction of rotation,
    and sing_rot=1 means rotate the Q,U in detector coordinate to Q,U in telescope coordinate.
    """
    if len(aman[Q_name].shape) == 1:
        Q_new = aman[Q_name] * np.cos(2*aman.focal_plane.gamma) + sign_rot * aman[U_name] * np.sin(2*aman.focal_plane.gamma)
        U_new = - sign_rot * aman[Q_name] * np.sin(2*aman.focal_plane.gamma) + aman[U_name] * np.cos(2*aman.focal_plane.gamma)
    elif len(aman[Q_name].shape) == 2:
        Q_new = aman[Q_name] * np.cos(2*aman.focal_plane.gamma[:, np.newaxis]) + sign_rot * aman[U_name] * np.sin(2*aman.focal_plane.gamma[:, np.newaxis])
        U_new = - sign_rot * aman[Q_name] * np.sin(2*aman.focal_plane.gamma[:, np.newaxis]) + aman[U_name] * np.cos(2*aman.focal_plane.gamma[:, np.newaxis])
    return Q_new, U_new
 
def wrap_fp_coords(aman):
    theta_fp = np.arcsin(np.sqrt(aman.focal_plane.xi**2 + aman.focal_plane.eta**2))
    phi_fp = np.arctan2(-aman.focal_plane.xi, -aman.focal_plane.eta)
    aman.focal_plane.wrap('theta_fp', theta_fp, [(0, 'dets'),])
    aman.focal_plane.wrap('phi_fp', phi_fp, [(0, 'dets'),])
    return


hwpss_param_names = ['hwpss4f_Q_tele_val', 'hwpss4f_Q_tele_err', 'hwpss4f_U_tele_val', 'hwpss4f_U_tele_err', 
                     'hwpss4f_P_tele_val', 'hwpss4f_P_tele_err', 'hwpss4f_theta_tele_val', 'hwpss4f_theta_tele_err',
                     'hwpss4f_P_radialQ_val', 'hwpss4f_P_radialQ_err', 'hwpss4f_P_radialU_val', 'hwpss4f_P_radialU_err']
leakage_param_names = ['leakage4f_Q_tele_val', 'leakage4f_Q_tele_err', 'leakage4f_U_tele_val', 'leakage4f_U_tele_err', 
                       'leakage4f_P_tele_val', 'leakage4f_P_tele_err', 'leakage4f_theta_tele_val', 'leakage4f_theta_tele_err',
                       'leakage4f_P_radialQ_val', 'leakage4f_P_radialQ_err', 'leakage4f_P_radialU_val', 'leakage4f_P_radialU_err']

def get_4f_stats(aman, wrap_stats=True):
    phi_fp = aman.focal_plane.phi_fp
    # hwpss4f parameters
    demodQ_tele, demodU_tele = rotate_QU(aman, Q_name='demodQ', U_name='demodU')
    aman.wrap('demodQ_tele', demodQ_tele, [(0, 'dets'), (1, 'samps'),])
    aman.wrap('demodU_tele', demodU_tele, [(0, 'dets'), (1, 'samps'),])

    demodQ_tele_ma = np.ma.masked_array(data=aman.demodQ_tele, mask=aman.flags.glitches.mask())
    demodU_tele_ma = np.ma.masked_array(data=aman.demodU_tele, mask=aman.flags.glitches.mask())
    hwpss4f_Q_tele_val = demodQ_tele_ma.mean(axis=1)
    hwpss4f_Q_tele_err = demodQ_tele_ma.std(axis=1)
    hwpss4f_U_tele_val = demodU_tele_ma.mean(axis=1)
    hwpss4f_U_tele_err = demodU_tele_ma.std(axis=1)
    hwpss4f_P_tele_val = np.sqrt(hwpss4f_Q_tele_val**2 + hwpss4f_U_tele_val**2)
    hwpss4f_P_tele_err = np.sqrt((hwpss4f_Q_tele_val / hwpss4f_P_tele_val * hwpss4f_Q_tele_err)**2 + (hwpss4f_U_tele_val / hwpss4f_P_tele_val * hwpss4f_U_tele_err)**2)
    hwpss4f_theta_tele_val = 0.5 * np.arctan2(hwpss4f_U_tele_val, hwpss4f_Q_tele_val)
    hwpss4f_theta_tele_err= 0.5*np.sqrt(((-hwpss4f_U_tele_val / (hwpss4f_Q_tele_val**2 + hwpss4f_U_tele_val**2)) * hwpss4f_Q_tele_err)**2 + \
                            ((hwpss4f_Q_tele_val / (hwpss4f_Q_tele_val**2 + hwpss4f_U_tele_val**2)) * hwpss4f_U_tele_err)**2)
    hwpss4f_P_radialQ_val = hwpss4f_Q_tele_val * np.cos(2*phi_fp) +  hwpss4f_U_tele_val * np.sin(2*phi_fp)
    hwpss4f_P_radialU_val = -hwpss4f_Q_tele_val * np.sin(2*phi_fp) +  hwpss4f_U_tele_val * np.cos(2*phi_fp)
    hwpss4f_P_radialQ_err = np.sqrt((np.cos(2 * phi_fp) * hwpss4f_Q_tele_err)**2 + (np.sin(2 * phi_fp) * hwpss4f_U_tele_err)**2)
    hwpss4f_P_radialU_err = np.sqrt((np.sin(2 * phi_fp) * hwpss4f_Q_tele_err)**2 + (np.cos(2 * phi_fp) * hwpss4f_U_tele_err)**2)

    # leakage4f parameters
    oman = t2pleakage.get_t2p_coeffs(aman, Q_sig_name='demodQ_tele', U_sig_name='demodU_tele', merge_stats=False, flag_name='glitches')
    leakage4f_Q_tele_val = oman.coeffsQ
    leakage4f_Q_tele_err = oman.errorsQ
    leakage4f_U_tele_val = oman.coeffsU
    leakage4f_U_tele_err = oman.errorsU
    leakage4f_P_tele_val = np.sqrt(leakage4f_Q_tele_val**2 + leakage4f_U_tele_val**2)
    leakage4f_P_tele_err = np.sqrt((leakage4f_Q_tele_val / leakage4f_P_tele_val * leakage4f_Q_tele_err)**2 + (leakage4f_U_tele_val / leakage4f_P_tele_val * leakage4f_U_tele_err)**2)
    leakage4f_theta_tele_val = 0.5 * np.arctan2(leakage4f_U_tele_val, leakage4f_Q_tele_val)
    leakage4f_theta_tele_err= 0.5*np.sqrt(((-leakage4f_U_tele_val / (leakage4f_Q_tele_val**2 + leakage4f_U_tele_val**2)) * leakage4f_Q_tele_err)**2 + \
                            ((leakage4f_Q_tele_val / (leakage4f_Q_tele_val**2 + leakage4f_U_tele_val**2)) * leakage4f_U_tele_err)**2)
    leakage4f_P_radialQ_val = leakage4f_Q_tele_val * np.cos(2 * phi_fp) + leakage4f_U_tele_val * np.sin(2 * phi_fp)
    leakage4f_P_radialU_val = -leakage4f_Q_tele_val * np.sin(2 * phi_fp) + leakage4f_U_tele_val * np.cos(2 * phi_fp)
    leakage4f_P_radialQ_err = np.sqrt((np.cos(2 * phi_fp) * leakage4f_Q_tele_err)**2 + (np.sin(2 * phi_fp) * leakage4f_U_tele_err)**2)
    leakage4f_P_radialU_err = np.sqrt((np.sin(2 * phi_fp) * leakage4f_Q_tele_err)**2 + (np.cos(2 * phi_fp) * leakage4f_U_tele_err)**2)
    
    if wrap_stats:
        hwpss_params = [hwpss4f_Q_tele_val, hwpss4f_Q_tele_err, hwpss4f_U_tele_val, hwpss4f_U_tele_err, 
                        hwpss4f_P_tele_val, hwpss4f_P_tele_err, hwpss4f_theta_tele_val, hwpss4f_theta_tele_err,
                        hwpss4f_P_radialQ_val, hwpss4f_P_radialQ_err, hwpss4f_P_radialU_val, hwpss4f_P_radialU_err]

        leakage_params = [leakage4f_Q_tele_val, leakage4f_Q_tele_err, leakage4f_U_tele_val, leakage4f_U_tele_err, 
                          leakage4f_P_tele_val, leakage4f_P_tele_err, leakage4f_theta_tele_val, leakage4f_theta_tele_err,
                          leakage4f_P_radialQ_val, leakage4f_P_radialQ_err, leakage4f_P_radialU_val, leakage4f_P_radialU_err]

        for name, param in zip(hwpss_param_names, hwpss_params):
            aman.wrap(name, param, [(0, 'dets')])

        for name, param in zip(leakage_param_names, leakage_params):
            aman.wrap(name, param, [(0, 'dets')])
    return

def process_aman(aman, subtract_hwpss=False):
    wrap_fp_coords(aman)
    aman = tod_process(aman, subtract_hwpss=subtract_hwpss)
    get_4f_stats(aman, wrap_stats=True)
    return aman
    
def remove_tods(aman):
    to_be_removed = []
    for key, vals in aman._assignments.items():
        if 'samps' in vals:
            to_be_removed.append(key)
    for field in to_be_removed:
        aman.move(field, None)
    return

def save_aman(aman, save_dir, obs_id, ws, bandpass):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f'{obs_id}_{ws}_{bandpass}.hdf'
    filepath = os.path.join(save_dir, filename)
    aman.save(filepath, overwrite=True)
    print(f'saved: {filepath}')
    return


def process_onewafer(ctx, obs_id, ws, bandpass, debug=False, save=False, save_dir=None):
    print(f'loading data: obs_id={obs_id}, ws={ws}, bandpass={bandpass}')
    aman = load_data(ctx, obs_id=obs_id, ws=ws, bandpass=bandpass, debug=debug,
                      do_correct_iir=True,
                      do_rn_selection=True, 
                      do_calibration=True, 
                      do_correct_hwp=True)
    aman = process_aman(aman)
    if save:
        print('removing heavy tods')
        remove_tods(aman)
        assert save_dir is not None
        os.makedirs(save_dir, exist_ok=True)
        print('saving aman')
        save_aman(aman=aman, save_dir=save_dir, obs_id=obs_id, ws=ws, bandpass=bandpass)
        
    return aman

def process_oneobs(ctx, obs_id, bandpass, save_dir, combined_save_dir, debug=False):
    assert save_dir is not None
    
    wafer_slots = ['ws{}'.format(index) for index, bit in enumerate(obs_id[-7:]) if bit == '1']
    for ws in wafer_slots:
        process_onewafer(ctx, obs_id, ws, bandpass, debug=debug, save=True, save_dir=save_dir)
    
    combine_amans(ctx, obs_id, bandpass, dir_name=save_dir, save=True, save_dir=combined_save_dir)
    return
        
def combine_amans(ctx, obs_id, bandpass, dir_name, save=False, save_dir=None):
    param_names = hwpss_param_names + leakage_param_names
    aman_all = ctx.get_meta(obs_id)
    wrap_fp_coords(aman_all)
    for param_name in param_names:
        aman_all.wrap(param_name, np.nan*np.zeros(aman_all.dets.count, dtype='float32'), [(0, 'dets')])
        
    filenames = glob.glob(os.path.join(dir_name, f'*{obs_id}*{bandpass}.hdf'))
    filenames.sort()
    
    for filename in filenames[:]:
        aman = core.AxisManager.load(filename)
        for param_name in param_names:
            for di, det in enumerate(aman.dets.vals):
                di_all = np.where(aman_all.dets.vals == det)[0][0]
                aman_all[param_name][di_all] = aman[param_name][di]
    aman_all.restrict('dets', aman_all.dets.vals[~np.isnan(aman_all[param_names[0]])])
    
    if save==True:
        assert save_dir is not None
        os.makedirs(save_dir, exist_ok=True)
        aman_all.save(os.path.join(save_dir, f'{obs_id}_{bandpass}.hdf'), overwrite=True)
    return aman_all

def main():
    parser = argparse.ArgumentParser(description="Process some observations.")
    parser.add_argument('obs_id', type=str, help='Observation ID')
    parser.add_argument('bandpass', type=str, help='Bandpass')
    parser.add_argument('--debug', type=int, default=0, help='Debug flag (default: 0)')
    parser.add_argument('--wafer_slot', type=str, default='all', help='Wafer slot (default: all)')

    args = parser.parse_args()

    obs_id = args.obs_id
    bandpass = args.bandpass
    debug = args.debugkkkkkk
    if debug==0:
        debug = False
    wafer_slot = args.wafer_slot
    
    ctx_file = '/so/home/tterasaki/MF1/2407/hwpss/use_this.yaml'
    ctx = core.Context(ctx_file)
    save_dir = '/so/home/tterasaki/MF1/2407/hwpss/result01'
    combined_save_dir = '/so/home/tterasaki/MF1/2407/hwpss/result01_combined'
    if wafer_slot=='all':
        process_oneobs(ctx=ctx, obs_id=obs_id, bandpass=bandpass, save_dir=save_dir, combined_save_dir=combined_save_dir, debug=debug)
    else:
        process_onewafer(ctx=ctx, obs_id=obs_id, ws=wafer_slot, bandpass=bandpass, debug=debug, save=True, save_dir=save_dir)
        combine_amans(ctx, obs_id, bandpass, dir_name=save_dir, save=True, save_dir=combined_save_dir)
    return

if __name__ == '__main__':
    main()
    
# python /so/home/tterasaki/repos/soteralib/soteralib/hwpss/analyze_hwpss.py obs_1714176018_satp1_1111111 f150 --debug 50 --wafer_slot all
    
