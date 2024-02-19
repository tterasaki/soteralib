#### Functions to check maps ####
import h5py
import numpy as np

from pixell import enmap, enplot


################ Functions to load single det maps ################
def get_detslist_in_hdf(map_hdf):
    with h5py.File(map_hdf, 'r') as f:
        dets = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
    return dets

def readmap_in_hdf(map_hdf, det):
    with h5py.File(map_hdf, 'r') as f:
        mT = enmap.read_hdf(f[det])
    return mT

def plotmap_in_hdf(map_hdf, det, upgrade=5, **kargs):
    mT = readmap_in_hdf(map_hdf, det)
    mT[mT==0.] = np.nan
    plot = enplot.plot(enmap.upgrade(mT, upgrade), **kargs)
    enplot.show(plot)
    return

################ Functions to get radial profile #################
def get_center_of_mass(mT):
    total_mass = float(np.nansum(mT))
    dec_grid, ra_grid = mT.posmap()
    dec_mass_center = float(np.nansum(dec_grid*mT)) / total_mass
    ra_mass_center = float(np.nansum(ra_grid*mT)) / total_mass
    return dec_mass_center, ra_mass_center

def get_radial_profile(mT, origin='mass'):
    assert origin in ['mass', 'max', 'zero']
    if origin == 'mass':
        ref = get_center_of_mass(mT)
    elif origin == 'max':
        argmax_row = np.where(mT == np.nanmax(mT))[0][0]
        argmax_col = np.where(mT == np.nanmax(mT))[1][0]
        dec_grid, ra_grid = mT.posmap()
        ref = (dec_grid[argmax_row, argmax_col], ra_grid[argmax_row, argmax_col])
    elif origin == 'zero':
        ref = (0, 0)
    r = mT.modrmap(ref=ref)
    r = r.flatten()
    z = mT.flatten() 
    
    # remove nan values
    not_none = ~np.isnan(z)
    r = r[not_none]
    z = z[not_none]
    
    sorter = np.argsort(r)
    r = r[sorter]
    z = z[sorter]
    
    return r, z