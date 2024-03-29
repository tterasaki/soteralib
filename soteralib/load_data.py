import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sotodlib
from sotodlib import core
from sotodlib.io import load_smurf as ls
from sotodlib.io.load_smurf import Observations, Files, TuneSets, Tunes
from sotodlib.tod_ops.fft_ops import calc_psd, calc_wn
from sotodlib.io import g3tsmurf_utils, hk_utils
from sodetlib.operations.iv import IVAnalysis
from sotodlib.io.metadata import read_dataset
from sotodlib.io.load_book import load_smurf_npy_data, get_cal_obsids

pA_per_phi0 = 9e6
phase_to_pA = pA_per_phi0 / (2*np.pi)

#################################################################
########################## Level 2 ##############################
#################################################################
def get_obs_all_level2(telescope):
    archive_path=f'/so/level2-daq/{telescope}/timestreams/'
    db_path=f'/so/level2-daq/databases/{telescope}/g3tsmurf.db'
    meta_path=f'/so/level2-daq/{telescope}/smurf/'
    hk_db_path=f'/so/level2-daq/databases/{telescope}/g3hk.db'
    
    SMURF = ls.G3tSmurf(archive_path=archive_path, db_path=db_path, meta_path=meta_path, hk_db_path=hk_db_path)
    session = SMURF.Session()
    obs_all = session.query(Observations)
    return SMURF, session, obs_all
    
def load_data_level2(obs_id, telescope, slice_obs_files=slice(None,None), unit='pA',
                    bgmap_style='last', iv_style='last', biasstep_style='last', 
                    calc_PSD=True, load_acu=True, load_fake_focal_plane=True, 
                    load_pointing=False, pointing_hdf=None):
    SMURF, session, obs_all = get_obs_all_level2(telescope)
    
    obs = obs_all.filter(Observations.obs_id == obs_id).one()
    fs = [f.name.replace('data', 'so/level2-daq') for f in obs.files[slice_obs_files]]
    aman = ls.load_file(fs, archive=SMURF)
    

        
    # bgmap
    if bgmap_style == 'last':
        bgmap_file = g3tsmurf_utils.get_next_bg_map(obs_id, SMURF)
        wrap_bgmap_level2(aman, bgmap_file)
    elif bgmap_style == 'next':
        bgmap_file = g3tsmurf_utils.get_next_bg_map(obs_id, SMURF)
        wrap_bgmap_level2(aman, bgmap_file)
    else:
        pass
    
    # iv
    if iv_style == 'last':
        iv_file = g3tsmurf_utils.get_last_iv(obs_id, SMURF)
        wrap_iv_level2(aman, iv_file)
    elif iv_style == 'next':
        iv_file = g3tsmurf_utils.get_next_iv(obs_id, SMURF)
        wrap_iv_level2(aman, iv_file)
    else:
        pass
    
    # biasstep
    if biasstep_style == 'last':
        biasstep_file = g3tsmurf_utils.get_last_bias_step(obs_id, SMURF)
        wrap_biasstep_level2(aman, biasstep_file)
    elif biasstep_style == 'next':
        biasstep_file = g3tsmurf_utils.get_next_bias_step(obs_id, SMURF)
        wrap_biasstep_level2(aman, biasstep_file)
    else:
        pass
    
    if unit=='pA':
        aman.signal *= phase_to_pA
    elif unit=='pW':
        aman.signal *= phase_to_pA
        aman.signal /= - aman.biasstep.si[:, np.newaxis]
    else:
        raise(ValueError)
        
        
    # calculate PSD
    if calc_PSD:
        freqs, Pxx = calc_psd(aman, nperseg=200*60, merge=True)
        wn = calc_wn(aman, pxx=Pxx, freqs=freqs)
        aman.wrap('wn', wn, [(0, 'dets')])
        
    if load_acu:
        wrap_acu(aman, telescope)
        
    if load_fake_focal_plane:
        wrap_fake_focalplane(aman)
    elif load_pointing:
        wrap_fake_focalplane(aman)
        wrap_pointing(aman, pointing_hdf)
        
    return aman

def wrap_bgmap_level2(aman, bgmap_file):
    bgmap_dict = np.load(bgmap_file, allow_pickle=True).item()
    bgmap = bgmap_dict['bgmap']
    aman.wrap(name='bgmap', data=bgmap, axis_map=[(0,'dets')])
    
    # band allocation
    is90 = np.in1d(aman.bgmap, [0,1,4,5,8,9])
    is150 = np.in1d(aman.bgmap, [2,3,6,7,10,11])
    aman.wrap('is90', is90, [(0, 'dets')])
    aman.wrap('is150', is150, [(0, 'dets')])
    return

def wrap_iv_level2(aman, iv_file):
    iva = IVAnalysis.load(iv_file)
    bias_samps = core.IndexAxis('bias_samps')
    ivman = core.AxisManager(aman.dets, bias_samps)
    
    ivman.wrap('v_bias', iva.v_bias, [(0, core.IndexAxis('bias_samps'))])
    ivman.wrap('i_bias', iva.i_bias, [(0, 'bias_samps')])
    ivman.wrap('R', iva.R, [(0, 'dets'), (1, 'bias_samps')])
    ivman.wrap('i_tes', iva.i_tes, [(0, 'dets'), (1, 'bias_samps')])
    ivman.wrap('v_tes', iva.v_tes, [(0, 'dets'), (1, 'bias_samps')])
    ivman.wrap('p_tes', iva.p_tes, [(0, 'dets'), (1, 'bias_samps')])
    ivman.wrap('si', iva.si, [(0, 'dets'), (1, 'bias_samps')])
    ivman.wrap('p_sat', iva.p_sat, [(0, 'dets')])
    ivman.wrap('R_n', iva.R_n, [(0, 'dets')])
    aman.wrap('iv', ivman)
    return

def wrap_biasstep_level2(aman, biasstep_file):
    bias_step_ana = np.load(biasstep_file, allow_pickle=True).item()
    bsman = core.AxisManager(aman.dets, aman.bias_lines)
    bsman.wrap('R0', bias_step_ana['R0'], [(0, 'dets')])
    bsman.wrap('tau_eff', bias_step_ana['tau_eff'], [(0, 'dets')])
    bsman.wrap('Rfrac', bsman.R0/aman.iv.R_n, [(0, 'dets')])
    bsman.wrap('I0', bias_step_ana['I0'], [(0, 'dets')])
    bsman.wrap('V0', bias_step_ana['R0'] * bias_step_ana['I0'], [(0, 'dets')])
    bsman.wrap('si', bias_step_ana['Si'], [(0, 'dets')])
    bsman.wrap('dItes', bias_step_ana['dItes'], [(0, 'dets')])
    bsman.wrap('Vbias', bias_step_ana['Vbias'], [(0, 'bias_lines')])
    bsman.wrap('Ibias', bias_step_ana['Ibias'], [(0, 'bias_lines')])
    bsman.wrap('dVbias', bias_step_ana['dVbias'], [(0, 'bias_lines')])
    bsman.wrap('dIbias', bias_step_ana['dIbias'], [(0, 'bias_lines')])
    bsman.wrap('R_sh', bias_step_ana['meta']['R_sh'])
    dRtes = bsman.R_sh * ((bsman.Ibias[aman.bgmap] + bsman.dIbias[aman.bgmap]) / (bsman.I0 + bsman.dItes) - 1 ) - bsman.R0
    bsman.wrap('dRtes', dRtes, [(0, 'dets')])
    dVtes = ((bsman.Ibias[aman.bgmap] + bsman.dIbias[aman.bgmap]) * ((bsman.R0+bsman.dRtes)**-1 + bsman.R_sh**-1)**-1 
            - bsman.Ibias[aman.bgmap] * (bsman.R0**-1 + bsman.R_sh**-1)**-1)
    bsman.wrap('dVtes', dVtes, [(0, 'dets')])
    aman.wrap('biasstep', bsman)
    return
    
def wrap_acu(aman, telescope):
    acu_aman = hk_utils.get_detcosamp_hkaman(aman, alias=['az', 'el'],
                              fields = [f'{telescope}.acu.feeds.acu_udp_stream.Corrected_Azimuth',
                                        f'{telescope}.acu.feeds.acu_udp_stream.Corrected_Elevation'],
                              data_dir = f'/so/level2-daq/{telescope}/hk')
    az = acu_aman.acu.acu[0]
    el = acu_aman.acu.acu[1]
    
    bman = core.AxisManager(aman.samps)
    bman.wrap('az', np.deg2rad(az), [(0, 'samps')])
    bman.wrap('el', np.deg2rad(el), [(0, 'samps')])
    bman.wrap('roll', 0*bman.az, [(0, 'samps')])
    aman.wrap('boresight', bman)
    return

def wrap_fake_focalplane(aman):
    fp = core.AxisManager(aman.dets)
    for key, value in zip(['xi', 'eta', 'gamma'], [0, 0, 0]):
        fp.wrap_new(key, shape=('dets', ))[:] = value
    aman.wrap('focal_plane', fp)
    return
    
    
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

#################################################################
########################## Level 3 ##############################
#################################################################
def get_obsdict_level3(ctx_file, query_line=None, query_tags=None):
    """
    query_line_example = 'start_time > 1704200000 and type == "obs"'
    query_tags_example = ['jupiter=1', 'rising=1']
    """
    ctx = core.Context(ctx_file)
    if query_line is not None:
        if query_tags is None:
            
            obslist= ctx.obsdb.query(query_line)
        else:
            
            obslist= ctx.obsdb.query(query_line, query_tags)
    else:
        obslist= ctx.obsdb.query()
        
    oids = np.unique([o['obs_id'] for o in obslist])
    obs_dict = {}
    for o in oids:
        _obsdict = ctx.obsdb.get(o, tags=True)
        obs_dict[o] = _obsdict
    return obs_dict
    
def load_data_level3(obs_id, ctx_file, meta=None):
    ctx = core.Context(ctx_file)
    if meta is None:
        meta = ctx.get_meta(obs_id)
    aman = ctx.get_obs(meta)
    return aman
        
def wrap_iv_level3(ctx, obs_id, aman):
    iv_obsids = get_cal_obsids(ctx, obs_id, 'iv')
    if np.unique(aman.det_info.detset).shape[0] > 1:
        raise ValueError('Restrict aman to only one wafer')
    detset = aman.det_info.detset[0]
    oid = iv_obsids[detset]
    iva = load_smurf_npy_data(ctx, oid, 'iv')
    rtm_bit_to_volt = iva['meta']['rtm_bit_to_volt']
    pA_per_phi0 = iva['meta']['pA_per_phi0']

    bias_samps = core.IndexAxis('bias_samps')
    states = core.LabelAxis('states', np.array(['sc', 'norm', '90per_ptes']))
    ivman = core.AxisManager(aman.dets, bias_samps, states)
    ivman.wrap('v_bias', iva['v_bias'], [(0, 'bias_samps')])
    ivman.wrap('i_bias', iva['i_bias'], [(0, 'bias_samps')])

    ivman.wrap_new('R', ('dets', 'bias_samps'))
    ivman['R'] *= np.nan
    ivman.wrap_new('i_tes', ('dets', 'bias_samps'))
    ivman['i_tes'] *= np.nan
    ivman.wrap_new('v_tes', ('dets', 'bias_samps'))
    ivman['v_tes'] *= np.nan
    ivman.wrap_new('p_tes', ('dets', 'bias_samps'))
    ivman['p_tes'] *= np.nan
    ivman.wrap_new('s_i', ('dets', 'bias_samps'))
    ivman['s_i'] *= np.nan
    ivman.wrap_new('state_idxs', ('dets', 'states'), dtype='int')
    
    ivman.wrap_new('p_sat', ('dets',))
    ivman['p_sat'] *= np.nan
    ivman.wrap_new('R_n', ('dets',))
    ivman['R_n'] *= np.nan
    

    for i, rid in enumerate(aman.det_info.readout_id):
        ridx = np.where(
                (iva['bands'] == aman.det_info.smurf.band[i]) & (iva['channels'] == aman.det_info.smurf.channel[i])
            )[0]
        if not ridx: # Channel doesn't exist in IV analysis
            continue
        ridx = ridx[0]

        ivman['R'][i, :] = iva['R'][ridx]
        ivman['i_tes'][i, :] = iva['i_tes'][ridx]
        ivman['v_tes'][i, :] = iva['v_tes'][ridx]
        ivman['p_tes'][i, :] = iva['p_tes'][ridx]
        ivman['s_i'][i, :] = iva['si'][ridx]
        ivman['state_idxs'][i, :] = iva['idxs'][ridx]
        ivman['p_sat'][i] = iva['p_sat'][ridx]
        ivman['R_n'][i] = iva['R_n'][ridx]

    aman.wrap('ivman', ivman)
    return

def wrap_biasstep_level3(ctx, obs_id, aman):
    biasstep_obsids = get_cal_obsids(ctx, obs_id, 'bias_steps')
    if np.unique(aman.det_info.detset).shape[0] > 1:
        raise ValueError('Restrict aman to only one wafer')
    detset = aman.det_info.detset[0]
    oid = biasstep_obsids[detset]
    bsa = load_smurf_npy_data(ctx, oid, 'bias_step_analysis')
    rtm_bit_to_volt = bsa['meta']['rtm_bit_to_volt']
    pA_per_phi0 = bsa['meta']['pA_per_phi0']
    
    if 'bias_lines' not in aman._fields.keys():
        aman._axes['bias_lines'] = core.LabelAxis('bias_lines', 
                                                  np.array([f'b{bl:02}' for bl in range(12)]))

    bsman = core.AxisManager(aman.dets, aman.bias_lines)
    bsman.wrap_new('R0', ('dets',))
    bsman['R0'] *= np.nan
    bsman.wrap_new('I0', ('dets',))
    bsman['I0'] *= np.nan
    bsman.wrap_new('V0', ('dets',))
    bsman['V0'] *= np.nan
    bsman.wrap_new('s_i', ('dets',))
    bsman['s_i'] *= np.nan
    bsman.wrap_new('tau_eff', ('dets',))
    bsman['tau_eff'] *= np.nan
    bsman.wrap_new('dItes', ('dets',))
    bsman['dItes'] *= np.nan

    for i, rid in enumerate(aman.det_info.readout_id):
        ridx = np.where(
                    (bsa['bands'] == aman.det_info.smurf.band[i]) & \
                    (bsa['channels'] == aman.det_info.smurf.channel[i])
                )[0]
        if not ridx: # Channel doesn't exist in IV analysis
                continue
        ridx = ridx[0]
        bsman['R0'][i] = bsa['R0'][ridx]
        bsman['I0'][i] = bsa['I0'][ridx]
        bsman['V0'][i] = bsa['R0'][ridx] * bsa['I0'][ridx]
        bsman['s_i'][i] = bsa['Si'][ridx]
        bsman['tau_eff'][i] = bsa['tau_eff'][ridx]
        bsman['dItes'][i] = bsa['dItes'][ridx]

    bsman.wrap('Vbias', bsa['Vbias'][:12], [(0, 'bias_lines')])
    bsman.wrap('Ibias', bsa['Ibias'][:12], [(0, 'bias_lines')])
    bsman.wrap('dVbias', bsa['dVbias'][:12], [(0, 'bias_lines')])
    bsman.wrap('dIbias', bsa['dIbias'][:12], [(0, 'bias_lines')])
    bsman.wrap('R_sh', bsa['meta']['R_sh'])
    dRtes = bsman.R_sh * ((bsman.Ibias[aman.det_cal.bg] + bsman.dIbias[aman.det_cal.bg])\
                          / (bsman.I0 + bsman.dItes) - 1 ) - bsman.R0
    bsman.wrap('dRtes', dRtes, [(0, 'dets')])

    dVtes = ((bsman.Ibias[aman.det_cal.bg] + bsman.dIbias[aman.det_cal.bg]) * ((bsman.R0+bsman.dRtes)**-1 + bsman.R_sh**-1)**-1 
                - bsman.Ibias[aman.det_cal.bg] * (bsman.R0**-1 + bsman.R_sh**-1)**-1)
    bsman.wrap('dVtes', dVtes, [(0, 'dets')])
    aman.wrap('bsman', bsman)
    return