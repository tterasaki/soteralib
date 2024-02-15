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
                    calc_PSD=True, load_acu=True, load_fake_focal_plane=True):
    SMURF, session, obs_all = get_obs_all_level2(telescope)
    
    obs = obs_all.filter(Observations.obs_id == obs_id).one()
    fs = [f.name.replace('data', 'so/level2-daq') for f in obs.files[slice_obs_files]]
    aman = ls.load_file(fs, archive=SMURF)
    

        
    # bgmap
    if bgmap_style == 'last':
        bgmap_file = g3tsmurf_utils.get_next_bg_map(obs_id, SMURF)
        wrap_bgmap(aman, bgmap_file)
    elif bgmap_style == 'next':
        bgmap_file = g3tsmurf_utils.get_next_bg_map(obs_id, SMURF)
        wrap_bgmap(aman, bgmap_file)
    else:
        pass
    
    # iv
    if iv_style == 'last':
        iv_file = g3tsmurf_utils.get_last_iv(obs_id, SMURF)
        wrap_iv(aman, iv_file)
    elif iv_style == 'next':
        iv_file = g3tsmurf_utils.get_next_iv(obs_id, SMURF)
        wrap_iv(aman, iv_file)
    else:
        pass
    
    # biasstep
    if biasstep_style == 'last':
        biasstep_file = g3tsmurf_utils.get_last_bias_step(obs_id, SMURF)
        wrap_biasstep(aman, biasstep_file)
    elif biasstep_style == 'next':
        biasstep_file = g3tsmurf_utils.get_next_bias_step(obs_id, SMURF)
        wrap_biasstep(aman, biasstep_file)
    else:
        pass
    
    if unit=='pA':
        aman.signal *= phase_to_pA
    elif unit=='pW':
        aman.signal *= phase_to_pA
        aman.signal /= - aman.biasstep.si[:, np.newaxis]
        
        
    # calculate PSD
    if calc_PSD:
        freqs, Pxx = calc_psd(aman, nperseg=200*60, merge=True)
        wn = calc_wn(aman, pxx=Pxx, freqs=freqs)
        aman.wrap('wn', wn, [(0, 'dets')])
        
    if load_acu:
        wrap_acu(aman, telescope)
        
    if load_fake_focal_plane:
        wrap_fake_focalplane(aman)
        
    return aman

def wrap_bgmap(aman, bgmap_file):
    bgmap_dict = np.load(bgmap_file, allow_pickle=True).item()
    bgmap = bgmap_dict['bgmap']
    aman.wrap(name='bgmap', data=bgmap, axis_map=[(0,'dets')])
    
    # band allocation
    is90 = np.in1d(aman.bgmap, [0,1,4,5,8,9])
    is150 = np.in1d(aman.bgmap, [2,3,6,7,10,11])
    aman.wrap('is90', is90, [(0, 'dets')])
    aman.wrap('is150', is150, [(0, 'dets')])
    return

def wrap_iv(aman, iv_file):
    iva = IVAnalysis.load(iv_file)
    ivman = core.AxisManager(aman.dets)
    ivman.wrap('v_bias', iva.v_bias, [(0, core.IndexAxis('v_bias'))])
    ivman.wrap('i_bias', iva.i_bias, [(0, 'v_bias')])
    ivman.wrap('R', iva.R, [(0, 'dets'), (1, 'v_bias')])
    ivman.wrap('i_tes', iva.i_tes, [(0, 'dets'), (1, 'v_bias')])
    ivman.wrap('v_tes', iva.v_tes, [(0, 'dets'), (1, 'v_bias')])
    ivman.wrap('p_tes', iva.p_tes, [(0, 'dets'), (1, 'v_bias')])
    ivman.wrap('si', iva.si, [(0, 'dets'), (1, 'v_bias')])
    ivman.wrap('p_sat', iva.p_sat, [(0, 'dets')])
    ivman.wrap('R_n', iva.R_n, [(0, 'dets')])
    aman.wrap('iv', ivman)
    return

def wrap_biasstep(aman, biasstep_file):
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
    

#################################################################
########################## Level 3 ##############################
#################################################################
def get_obsdict(ctx_file, query_line=None, query_tags=None):
    
    ctx = core.Context(ctx_file)
    if query_line is not None:
        if query_tags is None:
            #query_line_example = 'start_time > 1704200000 and type == "obs"'
            obslist= ctx.obsdb.query(query_line)
        else:
            #query_tags_example = ['jupiter=1', 'rising=1']
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
        
    