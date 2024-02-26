import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from sotodlib import core
from sotodlib import sim_hardware
from toast.instrument_coords import quat_to_xieta
from so3g.proj import quat
from pixell import enmap,enplot,utils

def get_so_focalplane_aman():
    nominal_hw = sim_hardware.sim_nominal()
    sim_hardware.sim_detectors_toast(nominal_hw, 'SAT1')
    dets = np.array(list(nominal_hw.data['detectors'].keys()), dtype='U30')
    
    bands = np.array([odict['band'][4:] for odict in nominal_hw.data['detectors'].values()], dtype='U4')
    wafer_slots = np.array([odict['wafer_slot'] for odict in nominal_hw.data['detectors'].values()])
    qs = [odict['quat'] for odict in nominal_hw.data['detectors'].values()]

    xi_eta_gammas = np.array([quat_to_xieta(q) for q in qs])
    xis,etas,gammas = xi_eta_gammas[:, 0], xi_eta_gammas[:, 1], xi_eta_gammas[:, 2]
    
    aman = core.AxisManager(core.LabelAxis('dets', dets))
    aman.wrap('band', bands, [(0, 'dets')])
    aman.wrap('wafer_slot', wafer_slots, [(0, 'dets')])

    focalplane = core.AxisManager(aman.dets)
    focalplane.wrap('xi', xis, [(0, 'dets')])
    focalplane.wrap('eta', etas, [(0, 'dets')])
    focalplane.wrap('gamma', gammas, [(0, 'dets')])
    aman.wrap('focalplane', focalplane)
    return aman

def get_zero_map(res=1*utils.degree, hemi=True, radius_deg=None):
    if radius_deg is None:
        if hemi:
            box = np.array([[0,180], [90,-180]]) * utils.degree
            shape,wcs = enmap.geometry(pos=box,res=res, proj='car')
        else:
            shape,wcs = enmap.fullsky_geometry(res=res, proj='car')
    else:
        box = np.array([[90-radius_deg,180], [90,-180]]) * utils.degree
        shape,wcs = enmap.geometry(pos=box,res=res, proj='car')
            
    zmap = enmap.zeros(shape=shape, wcs=wcs)
    return zmap

def get_beam_func_gauss(zmap, xieta_center, peak, sigma):
    xi_center, eta_center = xieta_center
    ra_center, dec_center, _ = quat.decompose_lonlat(quat.rotation_xieta(xi_center, eta_center))
    modrmap = zmap.modrmap(ref=(dec_center, ra_center))
    
    beam = np.exp( - modrmap**2 / (2*sigma**2))
    beam *= peak/beam.max()
    return beam

def get_ave_beam_with_ghost(aman, zmap, 
                            main_peak=1, main_sigma=0.3*utils.degree,
                            ghost_peak=0.1, ghost_sigma=0.5*utils.degree):
    
    main_beam = aman.dets.count * get_beam_func_gauss(zmap, (0,0), 
                                                      peak=main_peak, sigma=main_sigma)
    ghost_beam = zmap.copy()
    
    if ghost_peak > 0.:
        for di, det in enumerate(tqdm(aman.dets.vals)):
            xi_ghost = -2 * aman.focalplane.xi[di]
            eta_ghost = -2 * aman.focalplane.eta[di]
            ghost_beam += get_beam_func_gauss(zmap, (xi_ghost, eta_ghost), 
                                              peak=ghost_peak, sigma=ghost_sigma)
        beam = (main_beam + ghost_beam)/aman.dets.count
    else:
        beam = main_beam/aman.dets.count
        
    return beam


