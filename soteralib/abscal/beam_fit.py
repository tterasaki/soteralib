import os
import numpy as np
from scipy.optimize import curve_fit
from pixell import enmap, utils
from so3g.proj import coords, quat
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import re

def radec2xieta(radec, radec_center):
    """
    Rotate beam map to xieta coordinate.
    
    Args:
        radec (np.array of (2, pixels)): radec of map
        radec_center (list): (ra, dec) of center pixel.

    Returns:
        np.array of (2, pixels): xieta of pixels.
    """
    ra, dec = radec
    ra0, dec0 = radec_center
    q0 = quat.rotation_lonlat(lon=ra0, lat=dec0)
    q = quat.rotation_lonlat(lon=ra, lat=dec)
    xi, eta, _ = quat.decompose_xieta(~q0 * q)
    xieta = np.array((xi, eta))
    return xieta

def get_xieta(imap):
    """
    Get xieta of map.
    Args:
        imap (enmap): map
    Returns:
        np.array of (2, pixels): xieta of pixels.
    """
    dec, ra = imap.posmap().reshape([2, imap.posmap().shape[1]*imap.posmap().shape[2]])
    radec = (ra, dec)
    dec0, ra0 = imap.center()
    radec_center = (ra0, dec0)
    xieta = radec2xieta(radec, radec_center)
    return xieta

def trim_imap(imap, side):
    """
    Trim the map to the given side.
    Args:
        imap (enmap): map to be trimmed
        side (int): side to be trimmed
    Returns:
        enmap: trimmed map
    """
    submap_box = np.array([[-side/2, side/2], 
                         [side/2, -side/2]])
    submap = imap.submap(submap_box)
    return submap


def auto_trim(imap, shrink_factor=0.95):
    """
    Trim the map to remove zeros
    Args:
        imap (enmap): map to be trimmed
        shrink_factor (float): factor to shrink the map
    Returns:
        enmap: trimmed map
    """
    dec_side = np.abs(imap.box()[0][0] - imap.box()[1][0])
    ra_side = np.abs(imap.box()[0][1] - imap.box()[1][1])

    # take shorter side as the side to be trimmed
    if dec_side < ra_side:
        side = dec_side
    else:
        side = ra_side

    # shrink the side by the given factor until the map doesn't have zeros
    include_zero = True
    iter = 0
    while include_zero:
        #print(side)
        submap = trim_imap(imap, side)
        if np.size(submap) == np.count_nonzero(submap):
            include_zero = False
            break
        else:
            iter += 1
            #print('trimming')
            #print('side before:', side)
            side *= shrink_factor
            #print('side after:', side)
    return submap


def get_radial_profile(imap, ref='center'):
    """
    Get radial profile of the map.
    Args:
        imap (enmap): map of internsity 
        ref (str): reference point of the radial profile
    Returns:
        r_from_center_sorted (np.array of float): radial distance from the center (sorted)
        imap_sorted (np.array of float): map at the radial distance from the center (sorted)"""
    r_from_center_flatten = imap.mod
    rmap(ref=ref).reshape(imap.modrmap().size)
    imap_flatten = imap.reshape(np.size(imap))
    r_from_center_sorted = r_from_center_flatten[np.argsort(r_from_center_flatten)]
    imap_sorted = imap_flatten[np.argsort(r_from_center_flatten)]
    return r_from_center_sorted, imap_sorted

def gaussian2d(xieta, a, xi0, eta0, xi_sigma, eta_sigma, phi):
    """
    Gaussian2d beam model
    Args
    ------
    xieta: 2*N array of float
        containing xieta in the detector-centered coordinate system
    a: float
        amplitude of the Gaussian beam model
    xi0, eta0: float, float
        center position of the Gaussian beam model
    xi_sigma, eta_sigma, phi: float, float, float
        gaussian sigma along the xi, eta axis (rotated) and the rotation angle (in radians)
    Ouput:
    ------
    output: 1d array of float
        Time stream at sampling points given by xieta
    """
    xi, eta = xieta
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / xi_sigma ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / eta_sigma ** 2
    output = a * np.exp(xi_coef + eta_coef)
    return output

# SO design parameters of the beam
SO_BANDS         = np.array([27,    39,     93,     145,    225,    280]) * 1e9
SAT_DESIGN_SIGMA = np.array([91,	63,	    39,	    17,	    11,	    9]) * utils.arcmin / 2 / (2*np.log(2))**0.5
LAT_DESIGN_SIGMA = np.array([7.4,   5.1,    2.2,    1.4,	1,      0.9]) * utils.arcmin / 2 / (2*np.log(2))**0.5

def bandname2freq(bandname):
    freq = re.sub(r"\D", "", bandname)
    freq = float(freq) * 1e9
    return freq

def cov2corr(cov):
    D=np.diag(np.power(np.diag(cov),-0.5))
    corr=np.dot(np.dot(D,cov),D)
    return corr

class Fitter:
    def __init__(self, imap, model_name, telescope, freq_band, **kwargs):
        # map
        self.imap = imap
        self.xieta = get_xieta(self.imap)
        self.xi, self.eta = self.xieta[0], self.xieta[1]
        self.imap_flat_array = np.array(self.imap.reshape(self.imap.size))

        # information of the beam map
        self.telescope = telescope
        self.freq_band = freq_band
        
        
        
        # fit models and initial parameters
        self.fit_params_name = None
        self.params_init = None
        self.params_bounds = None

        # fit
        self.model_name = model_name
        self.fit_result = None
        self.opt_func = None
        self.fit_params = None
        self.fit_covar = None
        self.fit_corr = None
        self.fit_chi2 = None
        self.fit_redchi2 = None
        self.fit_nfree = None
        self.kwargs = kwargs
        
    def get_model_func(self):
        if self.model_name == 'gaussian2d':
            self.model_func = gaussian2d
        else:
            raise ValueError('Model name not supported')
    
    def get_isnan_mask(self):
        mask = np.isnan(self.imap_flat_array)
        self.isnan_mask = mask
        
        
    def get_issig_mask(self, sigr_deg=None):
        if sigr_deg is None:
            sigr = 5*self.sigma_init
        else:
            sigr = sigr_deg*coords.DEG
        mask = self.imap.modrmap() < sigr
        mask = np.array(mask.flatten())
        self.issig_mask = mask
    
    def set_init_params(self):
        xi0_init = 0.
        eta0_init = 0.
        phi_init = 0.
        a_init = np.nanmax(self.imap) - np.nanmin(self.imap)

        band_idx = np.argmin(np.abs(self.freq_band - SO_BANDS))
        if self.telescope == 'SAT':
            sigma_init = SAT_DESIGN_SIGMA[band_idx]
        elif self.telescope == 'LAT':
            sigma_init = LAT_DESIGN_SIGMA[band_idx]
        else:
            raise ValueError('Not supported name for telescope')

        if self.model_name == 'gaussian2d':
            self.fit_params_name = ['a', 'xi0', 'eta0', 'xi_sigma', 'eta_sigma', 'phi']
            self.sigma_init = sigma_init
            self.params_init = [a_init, xi0_init, eta0_init, sigma_init, sigma_init, phi_init]
            self.bounds = [(0, np.nanmin(self.xi),np.nanmin(self.eta), 0, 0, -np.pi),
                            (10*a_init, np.nanmax(self.xi), np.nanmax(self.eta), 10*sigma_init, 10*sigma_init, np.pi)]
        else:
            raise ValueError('Model name not supported')

    def set_sigma(self, manual_sigma=None):
        if manual_sigma is None:
            self.sigma = np.std(self.imap_flat_array) * np.ones_like(self.imap_flat_array) # This is not good. Need to fix it
            self.sigma = np.std(self.imap_flat_array[(~self.isnan_mask)&(~self.issig_mask)]) * np.ones_like(self.imap_flat_array)
        else:
            self.sigma = manual_sigma


    def fit(self):
        popt, pcov = curve_fit(self.model_func, 
                               self.xieta[:, ~self.isnan_mask], 
                               self.imap_flat_array[~self.isnan_mask],
                               sigma=self.sigma[~self.isnan_mask],
                               p0=self.params_init, bounds=self.bounds,
                              **self.kwargs)
        pcorr = cov2corr(pcov)
        
        self.fit_params = popt
        self.fit_params_dict = dict(zip(self.fit_params_name, self.fit_params))
        self.fit_errors_dict = dict(zip(self.fit_params_name, np.sqrt(np.diag(pcov))))
        self.fit_covar = pcov
        self.fit_corr = pcorr
        self.fit_errors = np.sqrt(np.diag(pcov))
        self.fit_chi2 = np.sum((self.imap_flat_array[~self.isnan_mask] - self.model_func(self.xieta[:, ~self.isnan_mask], *popt))**2 \
                               / self.sigma[~self.isnan_mask]**2)
        self.fit_nfree = len(self.imap_flat_array[~self.isnan_mask]) - len(self.fit_params)
        self.fit_redchi2 = self.fit_chi2 / self.fit_nfree
        model_map = self.model_func(self.xieta, *popt).reshape(self.imap.shape)
        model_map = enmap.enmap(model_map, self.imap.wcs)
        self.model_map = model_map
        self.residual_map = self.imap - self.model_map

        self.fit_result = { 'model_name': self.model_name,
                            'params_name': self.fit_params_name,
                            'params': self.fit_params,
                            'errors': self.fit_errors,
                            'params_dict': self.fit_params_dict,
                            'errors_dict': self.fit_errors_dict,
                            'covar': self.fit_covar,
                            'corr': self.fit_corr,
                            'chi2': self.fit_chi2,
                            'nfree': self.fit_nfree,
                            'redchi2': self.fit_redchi2,
                            }

    def plot_maps(self, fig=None, axes=None):
        if fig is None and axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
        extent = 60 * 180/np.pi * np.array([self.imap.box()[1][1], self.imap.box()[0][1], self.imap.box()[0][0], self.imap.box()[1][0]])
        mappable0 = axes[0].imshow(self.imap, origin='lower', extent=extent, label='data')
        model_label = self.model_name + '\n' + 'chi2/nfree = {:.2f}'.format(self.fit_redchi2)
        mappable1 = axes[1].imshow(self.model_map, origin='lower', extent=extent, label=model_label)
        mappable2 = axes[2].imshow(self.residual_map, origin='lower', extent=extent)

        fig.colorbar(mappable0, ax=axes[0], orientation='horizontal', label='det unit')
        fig.colorbar(mappable1, ax=axes[1], orientation='horizontal', label='det unit')
        fig.colorbar(mappable2, ax=axes[2], orientation='horizontal', label='det unit')
        for ax in axes:
            ax.set_xlabel('RA (arcmin)')
            ax.set_ylabel('Dec (arcmin)')
            
        axes[0].set_title('uncal map')
        axes[1].set_title(self.model_name +'\n(' + r' $\chi^2$'+'/dof={:.1f})'.format(self.fit_redchi2))
        axes[2].set_title('residual')
        return fig, axes

    def plot_along_axis(self, axis_along='x', fig=None, axes=None):
        if fig is None and axes is None:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,6), 
                        sharex=True, gridspec_kw={'height_ratios':[3,2], 'hspace':0.05})

        xi0 = self.fit_result['params_dict']['xi0']
        eta0 = self.fit_result['params_dict']['eta0']
        ypix0, xpix0 = enmap.sky2pix(self.imap.shape, self.imap.wcs, [eta0, xi0])
        ypix0, xpix0 = int(ypix0), int(xpix0)
        
        if axis_along == 'x':
            # profile along x-axis
            imap_cross_section_axis = 180/np.pi * 60 * self.imap.posmap()[1][ypix0,:]
            imap_cross_section_along_axis = self.imap[ypix0,:]
            model_map_cross_section_axis = 180/np.pi * 60 * self.model_map.posmap()[1][ypix0,:]
            model_map_cross_section_along_axis = self.model_map[ypix0,:]
            xlabel = 'x (arcmin)'
            ylabel = 'output (detector unit)'
            title = 'profile along x-axis'
        elif axis_along == 'y':
            # profile along y-axis
            imap_cross_section_axis =  180/np.pi * 60 * self.imap.posmap()[0][:,xpix0]
            imap_cross_section_along_axis = self.imap[:,xpix0]
            model_map_cross_section_axis = 180/np.pi * 60 *  self.model_map.posmap()[0][:,xpix0]
            model_map_cross_section_along_axis = self.model_map[:,xpix0]
            xlabel = 'y (arcmin)'
            ylabel = 'output (detector unit)'
            title = 'profile along y-axis'
        else:
            raise ValueError('axis_along should be either x or y')
    
        axes[0].plot(imap_cross_section_axis, imap_cross_section_along_axis, label='data')
        axes[0].plot(model_map_cross_section_axis, model_map_cross_section_along_axis, label='model')
        axes[1].plot(imap_cross_section_axis, imap_cross_section_along_axis-model_map_cross_section_along_axis,
                    label='data - model')
        axes[0].set_title(title)
        axes[0].set_ylabel(ylabel)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('data - model (detector unit)')
        axes[0].legend()
        axes[1].legend()
        return fig, axes

    def save_all(self, save_dir, prefix=''):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # dump the fit result
        fit_result_pkl_name = '{}_fit_result.pkl'.format(prefix)
        with open(os.path.join(save_dir, fit_result_pkl_name), 'wb') as f:
            pkl.dump(self.fit_result, f)
        
        # dump the fit params
        df = pd.DataFrame(np.vstack([self.fit_params, self.fit_errors]),
                  columns=self.fit_params_name,
                  index=['value', 'error'])
        params_df_name = '{}_params.h5'.format(prefix)
        df.to_hdf(os.path.join(save_dir, params_df_name), 'params')

        # save the plots
        fig, axes = self.plot_maps()
        save_fig_name_map = '{}_fit_map.png'.format(prefix)
        plt.savefig(os.path.join(save_dir, save_fig_name_map))

        fig, axes = self.plot_along_axis(axis_along='x')
        save_fig_name_alongx = '{}_fit_alongx.png'.format(prefix)
        plt.savefig(os.path.join(save_dir, save_fig_name_alongx))

        fig, axes = self.plot_along_axis(axis_along='y')
        save_fig_name_alongy = '{}_fit_alongy.png'.format(prefix)
        plt.savefig(os.path.join(save_dir, save_fig_name_alongy))

    def main(self):
        self.set_init_params()
        self.get_isnan_mask()
        self.get_issig_mask()
        self.set_sigma()
        self.get_model_func()
        self.fit()
        return 