import os
import sys
from tqdm import tqdm
import argparse
import numpy as np
import scipy
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
    _ = tod_ops.flags.get_turnaround_flags(aman, t_buffer=2.0, truncate=True)
    
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

def rotate_QU(aman, Q_name, U_name, sign_rot=1):
    """
    
    """
    if len(aman[Q_name].shape) == 1:
        Q_new = aman[Q_name] * np.cos(2*aman.focal_plane.gamma) + sign_rot * aman[U_name] * np.sin(2*aman.focal_plane.gamma)
        U_new = - sign_rot * aman[Q_name] * np.sin(2*aman.focal_plane.gamma) + aman[U_name] * np.cos(2*aman.focal_plane.gamma)
    elif len(aman[Q_name].shape) == 2:
        Q_new = aman[Q_name] * np.cos(2*aman.focal_plane.gamma[:, np.newaxis]) + sign_rot * aman[U_name] * np.sin(2*aman.focal_plane.gamma[:, np.newaxis])
        U_new = - sign_rot * aman[Q_name] * np.sin(2*aman.focal_plane.gamma[:, np.newaxis]) + aman[U_name] * np.cos(2*aman.focal_plane.gamma[:, np.newaxis])
    return Q_new, U_new

def wrap_QU_tele(aman, sign_rot):
    """
    現状はPとして、polarization powerしか計算していないが、focalplane radial方向か、circumfencial方向に分割したものも計算して、Essinger のプロットを書きたい。
    
    """
    demodQ_tele, demodU_tele = rotate_QU(aman, Q_name='demodQ', U_name='demodU', sign_rot=sign_rot)
    demodQ_mean_tele, demodU_mean_tele = rotate_QU(aman, Q_name='demodQ_mean', U_name='demodU_mean', sign_rot=sign_rot)
    lambda_demodQ_tele, lambda_demodU_tele = rotate_QU(aman, Q_name='lambda_demodQ', U_name='lambda_demodU', sign_rot=sign_rot)
    
    aman.wrap('demodQ_tele', demodQ_tele, [(0, 'dets'), (1, 'samps')])
    aman.wrap('demodU_tele', demodU_tele, [(0, 'dets'), (1, 'samps')])
    aman.wrap('demodQ_mean_tele', demodQ_mean_tele, [(0, 'dets')])
    aman.wrap('demodU_mean_tele', demodU_mean_tele, [(0, 'dets')])
    aman.wrap('lambda_demodQ_tele', lambda_demodQ_tele, [(0, 'dets')])
    aman.wrap('lambda_demodU_tele', lambda_demodU_tele, [(0, 'dets')])
    return

def calc_wrap_psd(aman, signal_name, merge=False, merge_wn=False, merge_suffix=None):
    freqs, Pxx = calc_psd(aman, signal=aman[signal_name], nperseg=nperseg, merge=False)
    if merge:
        assert merge_suffix is not None
        if 'nusamps' not in list(aman._axes.keys()):
            nusamps = core.OffsetAxis('nusamps', len(freqs))
            aman.wrap_new('freqs', (nusamps, ))
            aman.freqs = freqs
        aman.wrap(f'Pxx_{merge_suffix}', Pxx, [(0, 'dets'), (1, 'nusamps')])
    if merge_wn:
        wn = calc_wn(aman, pxx=Pxx, freqs=freqs)
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

def calc_opteff_corr(aman, bandpass, signal_name='dsT', ds_factor=100):
    mask = ~aman.flags.invalid_pwv.mask()
    _x = aman.pwv_class[mask][::ds_factor]
    el_boresight = np.median(aman.boresight.el)
    
    if bandpass == 'f090':
        p_zenith_100eff_slope = p_zenith_100eff_slope_f090
    elif bandpass == 'f150':
        p_zenith_100eff_slope = p_zenith_100eff_slope_f150

    opt_eff = np.zeros(aman.dets.count, dtype=float)
    corr_coeff = np.zeros(aman.dets.count, dtype=float)
    el_det = np.zeros(aman.dets.count, dtype=float)

    for di, det in enumerate(aman.dets.vals):
        _el_det = el_boresight + aman.focal_plane.eta[di]
        
        try:
            _y = aman[signal_name][di][mask][::ds_factor]
            _slope = np.polyfit(_x, _y , deg=1)[0]
            _p_100eff_slope = p_zenith_100eff_slope / np.sin(_el_det)
        except Exception as e:
            print(e)
            _slope = np.nan
            _p_100eff_slope = np.nan

        el_det[di] = _el_det
        opt_eff[di] = _slope / _p_100eff_slope
        corr_coeff[di] = np.corrcoef(_x, _y)[0][1]

    aman.wrap('el_det', el_det, [(0, 'dets')])
    aman.wrap('opt_eff', opt_eff, [(0, 'dets')])
    aman.wrap('corr_coeff', corr_coeff, [(0, 'dets')])
    aman.restrict('dets', aman.dets.vals[corr_coeff>0.])
    return

def calc_wrap_psd(aman, signal_name, merge=False, merge_wn=False, merge_suffix=None):
    freqs, Pxx = calc_psd(aman, signal=aman[signal_name], nperseg=nperseg, merge=False)
    if merge:
        assert merge_suffix is not None
        if 'nusamps' not in list(aman._axes.keys()):
            nusamps = core.OffsetAxis('nusamps', len(freqs))
            aman.wrap_new('freqs', (nusamps, ))
            aman.freqs = freqs
        aman.wrap(f'Pxx_{merge_suffix}', Pxx, [(0, 'dets'), (1, 'nusamps')])
    if merge_wn:
        wn = calc_wn(aman, pxx=Pxx, freqs=freqs)
        _ = aman.wrap(f'wn_{merge_suffix}', wn, [(0, 'dets')])
    return  freqs, Pxx

def tod_process(aman, subtract_hwpss=False):
    # get hwpss
    print('getting hwpss')
    _ = hwp.get_hwpss(aman)
    
    print('remove invalid pwv')
    remove_invalid_pwv(aman)
    
    print('detrend')
    tod_ops.detrend_tod(aman, method='median')
    
    if subtract_hwpss:
        print('subtract hwpss')
        hwp.subtract_hwpss(aman)
        aman.signal = aman.hwpss_remove
        aman.move('hwpss_remove', None)
        
    tod_ops.apodize_cosine(aman, apodize_samps=10000)
    calc_wrap_psd(aman, signal_name='signal', merge=True, merge_wn=True, merge_suffix='signal')
    
    print('demod')
    hwp.demod_tod(aman)
    calc_wrap_psd(aman, signal_name='dsT', merge=True, merge_wn=True, merge_suffix='dsT')
    calc_wrap_psd(aman, signal_name='demodQ', merge=True, merge_wn=True, merge_suffix='demodQ')
    calc_wrap_psd(aman, signal_name='demodU', merge=True, merge_wn=True, merge_suffix='demodU')
    aman.restrict('samps', (aman.samps.offset+10000, aman.samps.offset + aman.samps.count-10000))
    
    print('ptp cuts')
    ptp_cuts(aman, signal_name='dsT', kurtosis_threshold=2., print_state=False)
    aman.wrap('pwv_median', np.nanmedian(aman.pwv_class))
    fhwp = -(np.sum(np.diff(np.unwrap(aman.hwp_angle))) /(aman.timestamps[-1] - aman.timestamps[0])) / (2 * np.pi)
    aman.wrap('fhwp', fhwp)
    
    return aman
    
def get_glitches(aman):
    glitches_T = tod_ops.flags.get_glitch_flags(aman, signal_name='dsT', merge=True, name='glitches_T')
    glitches_Q = tod_ops.flags.get_glitch_flags(aman, signal_name='demodQ', merge=True, name='glitches_Q')
    glitches_U = tod_ops.flags.get_glitch_flags(aman, signal_name='demodU', merge=True, name='glitches_U')
    aman.flags.reduce(flags=['glitches_T', 'glitches_Q', 'glitches_U'], method='union', wrap=True, new_flag='glitches', remove_reduced=True)
    return

def leakage_model(dT, AQ, AU, lamQ, lamU):
    return AQ + lamQ*dT + 1.j*(AU + lamU*dT)

def get_corr(aman, mask=None, ds_factor=100):
    """
    statistic errorをつけたい。
    具体的には
    * dsTをのホワイトノイズレベルからsigmaの値を取得、dsTのエラーバーとして使う
    * demodQ, demodUについても同様にホワイトノイズレベルからsigmaの値を取得
    * それらを用いてleast-chi-square fittingにしたい。
    """
    if mask is None:
        mask = np.ones_like(aman.dsT, dtype='bool')
        
    A_Q_array = []
    A_U_array = []
    A_P_array = []
    lambda_Q_array = []
    lambda_U_array = []
    lambda_P_array = []
    
    for di, det in enumerate(tqdm(aman.dets.vals[:])):
        x = aman.dsT[di][mask[di]][::ds_factor]
        yQ = aman.demodQ[di][mask[di]][::ds_factor]
        yU = aman.demodU[di][mask[di]][::ds_factor]
        
        model = Model(leakage_model, independent_vars=['dT'])
        params = model.make_params(AQ=np.median(yQ), AU=np.median(yU),
                                   lamQ=0., lamU=0.)
        result = model.fit(yQ+1j*yU, params, dT=x)
        A_Q = result.params['AQ'].value
        A_U = result.params['AU'].value
        A_P = np.sqrt(A_Q**2 + A_U**2)
        lambda_Q = result.params['lamQ'].value
        lambda_U = result.params['lamU'].value
        lambda_P = np.sqrt(lambda_Q**2 + lambda_U**2)
        
        A_Q_array.append(A_Q)
        A_U_array.append(A_U)
        A_P_array.append(A_P)
        lambda_Q_array.append(lambda_Q)
        lambda_U_array.append(lambda_U)
        lambda_P_array.append(lambda_P)
    
    A_Q_array = np.array(A_Q_array)
    A_U_array = np.array(A_U_array)
    A_P_array = np.array(A_P_array)
    
    lambda_Q_array = np.array(lambda_Q_array)
    lambda_U_array = np.array(lambda_U_array)
    lambda_P_array = np.array(lambda_P_array)
    
    return A_Q_array, A_U_array, A_P_array, lambda_Q_array, lambda_U_array, lambda_P_array

def wrap_theta(aman):
    theta_pol_mean = 0.5*np.arctan2(aman.demodU_mean, aman.demodQ_mean)
    aman.wrap('theta_pol_mean', theta_pol_mean, [(0, 'dets')])
    theta_lambda = 0.5*np.arctan2(aman.lambda_demodU, aman.lambda_demodQ)
    aman.wrap('theta_lambda', theta_lambda, [(0, 'dets')])
    
    theta_pol_mean_tele = 0.5*np.arctan2(aman.demodU_mean_tele, aman.demodQ_mean_tele)
    aman.wrap('theta_pol_mean_tele', theta_pol_mean_tele, [(0, 'dets')])
    theta_lambda_tele = 0.5*np.arctan2(aman.lambda_demodU_tele, aman.lambda_demodQ_tele)
    aman.wrap('theta_lambda_tele', theta_lambda_tele, [(0, 'dets')])
    return

def process_aman(aman, subtract_hwpss=False):
    # demod
    print('tod processing')
    aman = tod_process(aman, subtract_hwpss=subtract_hwpss)
    
    # glitches
    print('glitch detection')
    get_glitches(aman)
    mask = ~aman.flags.glitches.mask()
    
    # get stats
    print('gettig correlation lambda')
    demodQ_mean, demodU_mean, demodP_mean, lambda_demodQ, lambda_demodU, lambda_demodP = get_corr(aman, mask=mask)
    
    aman.wrap('demodQ_mean', demodQ_mean, [(0, 'dets')])
    aman.wrap('demodU_mean', demodU_mean, [(0, 'dets')])
    aman.wrap('demodP_mean', demodP_mean, [(0, 'dets')])
    aman.wrap('lambda_demodQ', lambda_demodQ, [(0, 'dets')])
    aman.wrap('lambda_demodU', lambda_demodU, [(0, 'dets')])
    aman.wrap('lambda_demodP', lambda_demodP, [(0, 'dets')])
    
    wrap_QU_tele(aman, sign_rot=1)
    wrap_theta(aman)
    return aman
    
def process_one_obs(ctx, obs_id, ws, bandpass, subtract_hwpss, sign_rot, debug=False):
    print('loading data')
    aman = load_data(ctx, obs_id=obs_id, ws=ws, bandpass=bandpass, debug=debug,
                      do_correct_iir=True,
                      do_rn_selection=True, 
                      do_calibration=True, 
                      do_correct_hwp=True)
    aman = process_aman(aman)
    return aman

def remove_heavy_fields(aman):
    print('Removing heavy signal')
    _fields = ['timestamps', 'ancil', 'biases', 'primary', 'iir_params', 'hwp_angle',
               'flags', 'obs_info', 'det_cal', 'det_info', 'hwpss_model', 'dsT', 'demodQ', 'demodU',
              'hwp_solution', 'demodQ_tele', 'demodU_tele', 'boresight', 'pwv_class', 'mk_stage']
    
    for _field in _fields:
        aman.move(_field, None)
    return

def save_aman(aman, save_dir, obs_id, ws, bandpass):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f'{obs_id}_{ws}_{bandpass}.hdf'
    filepath = os.path.join(save_dir, filename)
    aman.save(filepath, overwrite=True)
    print(f'saved: {filepath}')
    return







# if __name__ == '__main__':
#     ctx_file = '/global/homes/t/ttera/scratch/site_analysis/code_dev/hk2meta_satp3/use_this_with_l3hk.yaml'
#     ctx = core.Context(ctx_file)
#     import sys
#     min_ctime = int(sys.argv[1])
#     max_ctime = int(sys.argv[2])
#     query_line = f'start_time > {min_ctime} and start_time < {max_ctime} and type == "obs" and subtype == "cmb"'
#     obslist = ctx.obsdb.query(query_line)
#     obs_ids = np.unique([o['obs_id'] for o in obslist])
#     debug = False
    
#     ###### test01    
#     save_homedir = '/global/homes/t/ttera/scratch/site_analysis/MF2/2404/gain_investigation/results_cmb_01'
#     subtract_hwpss = False
    
#     for obs_id in obs_ids[:]:
#         wafer_slots = ['ws{}'.format(index) for index, bit in enumerate(obs_id[-7:]) if bit == '1']
#         for ws in wafer_slots[:]:
#             for bandpass in ['f090', 'f150']:
#                 try:
#                     filename = f'{obs_id}_{ws}_{bandpass}.hdf'
#                     print(f'{obs_id}, {ws}, {bandpass}')
#                     hwpss_aman, pwv_aman = process_one_obs(ctx=ctx, obs_id=obs_id, ws=ws, bandpass=bandpass, 
#                                                           subtract_hwpss=subtract_hwpss, sign_rot=1, debug=debug)

#                     remove_heavy_fields(hwpss_aman)

#                     save_dir = os.path.join(save_homedir, 'hwpss_stats_light')
#                     os.makedirs(save_dir, exist_ok=True)
#                     save_aman(hwpss_aman, save_dir=save_dir, 
#                               obs_id=obs_id, ws=ws, bandpass=bandpass)

#                     save_dir = os.path.join(save_homedir, 'pwv_stats')
#                     os.makedirs(save_dir, exist_ok=True)
#                     save_aman(pwv_aman, save_dir=save_dir,
#                               obs_id=obs_id, ws=ws, bandpass=bandpass)

#                     # save hwpss quiver
#                     save_dir = os.path.join(save_homedir, 'plot_stat_4f')
#                     os.makedirs(save_dir, exist_ok=True)
#                     fig, ax = plot_HWPSS_quiver(hwpss_aman, quiver_scale=5**2, z_scale=1e3, 
#                                                 colorbar_label='HWPSS 4f DC [nW]',
#                                                 theta_name='theta_pol_mean_tele', P_name='demodP_mean')
#                     ax.set_title(f'{obs_id},{ws},{bandpass}, pwv={hwpss_aman.pwv_median:.2f}[mm], fhwp={hwpss_aman.fhwp:.2f}[Hz]')
#                     fig.tight_layout()
#                     plt.savefig(os.path.join(save_dir, f'{obs_id}_{ws}_{bandpass}.png'))
#                     plt.clf()
#                     plt.close()

#                     # save leakage quiver
#                     save_dir = os.path.join(save_homedir, 'plot_leakage_4f')
#                     os.makedirs(save_dir, exist_ok=True)
#                     fig, ax = plot_HWPSS_quiver(hwpss_aman, quiver_scale=5**2, z_scale=100, 
#                                                 colorbar_label='I-to-P coupling coefficient [%]',
#                                                 theta_name='theta_lambda_tele', P_name='lambda_demodP')
#                     ax.set_title(f'{obs_id},{ws},{bandpass}, pwv={hwpss_aman.pwv_median:.2f}[mm], fhwp={hwpss_aman.fhwp:.2f}[Hz]')
#                     fig.tight_layout()
#                     plt.savefig(os.path.join(save_dir, f'{obs_id}_{ws}_{bandpass}.png'))
#                     plt.clf()
#                     plt.close()

#                     # save opteff corrcoeff
#                     save_dir = os.path.join(save_homedir, 'plot_opteff_corrcoeff_hist')
#                     os.makedirs(save_dir, exist_ok=True)
#                     fig, ax = plot_opteff_corrcoeff_hist(pwv_aman)
#                     plt.savefig(os.path.join(save_dir, f'{obs_id}_{ws}_{bandpass}.png'))
#                     plt.clf()
#                     plt.close()

#                     # save plot_noise_vs_opteff
#                     save_dir = os.path.join(save_homedir, 'plot_noise_vs_opteff')
#                     os.makedirs(save_dir, exist_ok=True)
#                     fig, ax = plot_noise_vs_opteff(pwv_aman)
#                     plt.savefig(os.path.join(save_dir, f'{obs_id}_{ws}_{bandpass}.png'))
#                     plt.clf()
#                     plt.close()

#                     # save plot_pwv_vs_signal
#                     save_dir = os.path.join(save_homedir, 'plot_pwv_vs_signal')
#                     os.makedirs(save_dir, exist_ok=True)
#                     fig, ax = plot_pwv_vs_signal(pwv_aman)
#                     plt.savefig(os.path.join(save_dir, f'{obs_id}_{ws}_{bandpass}.png'))
#                     plt.clf()               
#                     plt.close()
#                 except:
#                     print(f'Error happened in {obs_id}_{ws}_{bandpass}')
#         with open(save_homedir+'_finished_obsids.txt', 'a') as f:
#             print(obs_id, file=f)