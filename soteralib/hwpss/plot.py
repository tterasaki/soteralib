import numpy as np
import matplotlib.pyplot as plt
import matplotlib
cm = matplotlib.colormaps.get_cmap('viridis')

# Plot the quiver of HWPSS
def plot_quiver(aman, figax=None, quiver_scale=8**2, z_scale=1, shrink=0.8,
                      P_name='hwpss4f_P_tele_val', theta_name='hwpss4f_theta_tele_val', label=None, colorbar_label=None):
    if figax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = figax
    quiveropts = dict(scale=quiver_scale, width=0.003, headlength=0, pivot='middle')
    
    #normを引数として渡せるようにする
    norm = matplotlib.colors.Normalize(vmin=np.percentile(z_scale * aman[P_name], 10), 
                                       vmax=np.percentile(z_scale * aman[P_name], 90), clip=False)
                                       
    im = ax.quiver(np.rad2deg(aman.focal_plane.xi), np.rad2deg(aman.focal_plane.eta),
                   np.sin(aman[theta_name]), 
                   np.cos(aman[theta_name]), 
                   z_scale * aman[P_name],
                   alpha=0.8, cmap=cm,
                   norm=norm,
                   label=label,
                   headaxislength=0, **quiveropts)

    fig.colorbar(im, ax=ax, orientation='horizontal', label=colorbar_label, shrink=shrink)
    ax.set_xlabel(r'xi [deg]', fontsize=15)
    ax.set_ylabel(r'eta [deg]', fontsize=15)
    ax.tick_params(axis = 'x', labelsize =15)
    ax.tick_params(axis = 'y', labelsize =15)
    ax.set_aspect(1)
    return fig, ax

def plot_data_model_comparison(aman, mode='hwpss4f'):
    assert mode in ['hwpss4f', 'leakage4f']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax in axes:
        ax.set_xlim(-19, 19)
        ax.set_ylim(-19, 19)

    if mode == 'hwpss4f':
        z_scale = 1e3
        ah_plot.plot_quiver(aman, figax=(fig, axes[0]),
                            P_name='hwpss4f_P_tele_val', theta_name='hwpss4f_theta_tele_val', 
                            z_scale=z_scale, colorbar_label='4f HWPSS Amplitude[pW]')

        fit_label = f'uniform_amp: {aman.hwpss4f_uniform_P_val*z_scale:.2f} [pA]\n'
        fit_label += f'radial_amp: {aman.hwpss4f_radial_P_pivot_val*z_scale:.2f} [pA]\n'
        fit_label += f'redchi2: {aman.hwpss4f_redchi2:.2f}'
        
        ah_plot.plot_quiver(aman, figax=(fig, axes[1]),
                            P_name='hwpss4f_P_total_model', theta_name='hwpss4f_theta_total_model',
                            z_scale=z_scale, colorbar_label='4f HWPSS Amplitude[pW]',
                           label=fit_label)
        axes[1].legend()
        axes[0].set_title(f'HWPSS data')
        axes[1].set_title(f'HWPSS fit')
        
    elif mode == 'leakage4f':
        z_scale = 1e2
        ah_plot.plot_quiver(aman, figax=(fig, axes[0]),
                            P_name='leakage4f_P_tele_val', theta_name='leakage4f_theta_tele_val', 
                            z_scale=z_scale, colorbar_label='4f Laakage Coefficient [%]')

        fit_label = f'uniform_coeff: {aman.leakage4f_uniform_P_val*z_scale:.2f} [%]\n'
        fit_label += f'radial_coeff: {aman.leakage4f_radial_P_pivot_val*z_scale:.2f} [%]\n'
        fit_label += f'redchi2: {aman.leakage4f_redchi2:.2f}'
        
        ah_plot.plot_quiver(aman, figax=(fig, axes[1]),
                            P_name='leakage4f_P_total_model', theta_name='leakage4f_theta_total_model',
                            z_scale=z_scale, colorbar_label='4f Laakage Coefficient [%]',
                           label=fit_label)
        axes[1].legend()

        
        axes[0].set_title(f'Leakage data')
        axes[1].set_title(f'Leakage fit')
            
    telescope = aman.obs_info.telescope
    bandpass = aman.det_info.wafer.bandpass[0]
    pwv = np.nanmedian(aman.pwv_class)
    fig.suptitle(f'{telescope}, {bandpass}, pwv={pwv:.2f}')
    fig.tight_layout()

def plot_hwpss_radial_profile(aman):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    wafer_slots = [f'ws{i}' for i in range(7)]
    for ws in wafer_slots:
        mask = aman.det_info.wafer_slot == ws
        ax[0].plot(180/np.pi*aman.focal_plane.theta_fp[mask], 1e3*aman.hwpss4f_P_tele_val[mask], '.', label=f'{ws}')
        ax[1].plot(180/np.pi*aman.focal_plane.theta_fp[mask], 1e3*aman.hwpss4f_P_total_model[mask], '.', label=f'{ws}')
        ax[2].plot(180/np.pi*aman.focal_plane.theta_fp[mask], 1e3*(aman.hwpss4f_P_tele_val[mask] - aman.hwpss4f_P_total_model[mask])
                   , '.', label=f'{ws}')

    ax[0].set_title('HWPSS data')
    ax[1].set_title('HWPSS model')
    ax[2].set_title('HWPSS residual')

    for a in ax[:]:
        a.set_xlabel('distance from fp center [deg]')
        a.set_ylabel('HWPSS amplitude [pW]')
        a.grid()
        a.legend()
    for a in ax[:-1]:
        a.set_ylim(-0.2, 20)
    ax[-1].set_ylim(-10, 10)

    fig.tight_layout()
    return

def plot_leakage_radial_profile(aman):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    wafer_slots = [f'ws{i}' for i in range(7)]
    for ws in wafer_slots:
        mask = aman.det_info.wafer_slot == ws
        ax[0].plot(180/np.pi*aman.focal_plane.theta_fp[mask], 100*aman.leakage4f_P_tele_val[mask], '.', label=f'{ws}')
        ax[1].plot(180/np.pi*aman.focal_plane.theta_fp[mask], 100*aman.leakage4f_P_total_model[mask], '.', label=f'{ws}')
        ax[2].plot(180/np.pi*aman.focal_plane.theta_fp[mask], 100*(aman.leakage4f_P_tele_val[mask] - aman.leakage4f_P_total_model[mask]),
                   '.', label=f'{ws}')

    ax[0].set_title('leakage data')
    ax[1].set_title('leakage model')
    ax[2].set_title('leakage residual')

    for a in ax:
        a.set_xlabel('distance from fp center [deg]')
        a.set_ylabel('leakge coeff [%]')

        a.grid()
        a.legend()
    for a in ax[:-1]:
        a.set_ylim(-0.2, 2)
    ax[-1].set_ylim(-1, 1)
    return

# Plot one detector (for debug)
def plot_onedet_timestreams(aman, di, demodQ_name='demodQ', demodU_name='demodU'):
    det = aman.dets.vals[di]
    alpha = 0.5
    fig, ax = plt.subplots(1,3, figsize=(12, 4))
    Dt = aman.timestamps - aman.timestamps[0]
    ax[0].plot(Dt, aman.dsT[di]-aman.dsT[di].mean(), color='black', label='delta T')
    ax[0].set_xlabel('time [sec]')
    ax[0].set_ylabel('[pW]')
    ax[0].legend()
    ymin, ymax = ax[0].get_ylim()
    
    ax[1].plot(Dt, aman[demodQ_name][di], color='red', label='Q')
    ax[1].plot(Dt, aman[demodU_name][di], color='blue', label='U')
    ax[1].set_ylim(ymin, ymax)
    ax[1].set_xlabel('time [sec]')
    ax[1].set_ylabel('[pW]')
    ax[1].legend()

    ax[2].plot(Dt, (aman.dsT[di]-aman.dsT[di].mean())*0.01, alpha=1., color='black', label='deltaT * 0.01')
    ax[2].plot(Dt, aman[demodQ_name][di]-aman[demodQ_name][di].mean(), alpha=alpha, color='red', label='Q - Q_mean')
    ax[2].plot(Dt, aman[demodU_name][di]-aman[demodU_name][di].mean(), alpha=alpha, color='blue', label='U - U_mean')
    ax[2].legend()
    ax[2].set_xlabel('time [sec]')
    ax[2].set_ylabel('[pW]')
    
    obs_id = aman.obs_info.obs_id
    ws = aman.det_info.wafer_slot[di]
    bandpass = aman.det_info.wafer.bandpass[di]
    ax[1].set_title(f'{obs_id}, {ws}, {bandpass}, {det}')
    fig.tight_layout()
    return fig, ax

def plot_onedet_correlation(aman, di, ds_factor=100,
                            demodQ_name='demodQ', demodU_name='demodU',
                            lambda_demodQ_name='lambda_demodQ', lambda_demodU_name='lambda_demodU'):
    det = aman.dets.vals[di]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    
    x = aman.dsT[di] - aman.dsT[di].mean()
    Dt = aman.timestamps - aman.timestamps[0]
    yQ = aman[demodQ_name][di] - aman[demodQ_name][di].mean()
    yU = aman[demodU_name][di] - aman[demodU_name][di].mean()
    yP = np.sqrt(aman[demodQ_name][di]**2 + aman[demodU_name][di]**2)
    
    x = x[::ds_factor]
    Dt = Dt[::ds_factor]
    yQ = yQ[::ds_factor]
    yU = yU[::ds_factor]
    yP = yP[::ds_factor]
    
    lambda_Q = aman[lambda_demodQ_name][di]
    lambda_U = aman[lambda_demodU_name][di]
    lambda_P = np.sqrt(lambda_Q**2 + lambda_U**2)
    
    im = ax[0].scatter(x, yQ, s=0.1, c=Dt, label=r'$\lambda_{Q}=$'+f'{lambda_Q*1e2:.1f} [%]')
    im = ax[1].scatter(x, yU, s=0.1, c=Dt, label=r'$\lambda_{U}=$'+f'{lambda_U*1e2:.1f} [%]')
    im = ax[2].scatter(x, yP, s=0.1, c=Dt, label=r'$\lambda_{P}=$'+f'{lambda_P*1e2:.1f} [%]')
    
    for _a, X in zip(ax, ['Q', 'U', 'P']):
        _a.grid()
        _a.set_xlabel('delta T [pW]')
        _a.set_ylabel(f'{X} - {X}_mean [pW]')
        _a.legend()
        
    obs_id = aman.obs_info.obs_id
    ws = aman.det_info.wafer_slot[di]
    bandpass = aman.det_info.wafer.bandpass[di]
    ax[1].set_title(f'{obs_id}, {ws}, {bandpass}, {det}')
    fig.tight_layout()
    return fig, ax

def plot_onedet_QUplane(aman, di, ds_factor=100,
                        demodQ_name='demodQ', demodU_name='demodU', 
                        demodQ_mean_name='demodQ_mean', demodU_mean_name='demodU_mean', theta_pol_mean_name='theta_pol_mean',
                        lambda_demodQ_name='lambda_demodQ', lambda_demodU_name='lambda_demodU', theta_lambda_name='theta_lambda'):
    mask = ~aman.flags.glitches.mask()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    x = aman.dsT[di][mask[di]]
    Dt = aman.timestamps[mask[di]] - aman.timestamps[0]
    yQ = aman[demodQ_name][di][mask[di]]
    yU = aman[demodU_name][di][mask[di]]

    x = x[::ds_factor]
    Dt = Dt[::ds_factor]
    yQ = yQ[::ds_factor]
    yU = yU[::ds_factor]

    lambda_Q = aman[lambda_demodQ_name][di]
    lambda_U = aman[lambda_demodU_name][di]
    label = r'$\lambda_{Q}=$'+f'{lambda_Q*1e2:.2f} [%]' + '\n' + r'$\lambda_{U}=$'+f'{lambda_U*1e2:.2f} [%]'
    im = ax[0].scatter(yQ, yU, s=0.5, c=x, label=label, cmap=cm)
    im = ax[1].scatter(yQ, yU, s=0.5, c=x, label=label, cmap=cm)
    fig.colorbar(im, ax=ax[1], shrink=0.8, label=r'$\Delta T$ [pW]')

    # rectangle
    xlim0, xlim1 = ax[0].get_xlim()
    ylim0, ylim1 = ax[0].get_ylim()
    limmax = np.max(np.abs([xlim0, xlim1, ylim0, ylim1]))*1.05
    r = patches.Rectangle( (xlim0,ylim0) , xlim1-xlim0, ylim1-ylim0, fill=False, edgecolor="black", linewidth=0.5)
    ax[1].add_patch(r)


    for _a in ax:
        # (zeropoint)-to-(mean point)
        _x = np.linspace(-2*aman[demodQ_mean_name][di], 2*aman[demodQ_mean_name][di], 10)
        _y = np.linspace(-2*aman[demodU_mean_name][di], 2*aman[demodU_mean_name][di], 10)

        label_zero2mean = '(zero point)-to-(mean point)' + '\n' + r'$\arctan2(U,Q)=$' + \
                            f'{2*np.rad2deg(aman[theta_pol_mean_name][di]):.1f} [deg]'
        _a.plot(_x, _y, ':', color='black', label=label_zero2mean)

        # linear fit
        label_linear_fit = 'linear fit' + '\n' + r'$\arctan2((dU/dT),(dQ/dT))=$' + \
                            f'{2*np.rad2deg(aman[theta_lambda_name][di]):.1f} [deg]'
        
        leakage_fitted = leakage_model(np.array([100*np.min(x), 100*np.max(x)]),
                              aman[demodQ_mean_name][di], aman[demodU_mean_name][di], 
                              aman[lambda_demodQ_name][di], aman[lambda_demodU_name][di],)
        
        _a.plot( leakage_fitted.real, leakage_fitted.imag, ':', color='red', label=label_linear_fit)

        _a.grid()
        _a.set_aspect(1)
        _a.set_xlabel('demodQ [pW]')
        _a.set_ylabel('demodU [pW]')
        _a.legend()

    ax[0].set_xlim(xlim0, xlim1)
    ax[0].set_ylim(ylim0, ylim1)
    ax[1].set_xlim(-limmax, limmax)
    ax[1].set_ylim(-limmax, limmax)
    
    obs_id = aman.obs_info.obs_id
    ws = aman.det_info.wafer_slot[di]
    bandpass = aman.det_info.wafer.bandpass[di]
    ax[0].set_title(f'{obs_id}, {ws}, {bandpass}, {det}')
    
    fig.tight_layout()
    return fig, ax



def plot_opteff_corrcoeff_hist(aman):
    obs_id = aman.obs_info.obs_id
    ws = aman.det_info.wafer_slot[0]
    bandpass = aman.det_info.wafer.bandpass[0]
    color = 'tab:blue' if bandpass=='f090' else 'tab:orange'
    
    fig, ax = plt.subplots(1,2, figsize=(9, 4))
    bins = np.linspace(0, 100, 50)
    _ = ax[0].hist(aman.opt_eff*100, bins=bins, color=color)
    ax[0].set_xlim(0, 100)
    ax[0].set_xlabel('optical efficiency [%]')
    ax[0].set_ylabel('count')
    bins = np.linspace(0, 1., 50)
    _ = ax[1].hist(aman.corr_coeff, bins=bins, color=color)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel('correlation coeefficient\n' + r'(pwv-vs-$\Delta T$)')
    ax[1].set_ylabel('count')
    fig.suptitle(f'{obs_id}, {ws}, {bandpass}, #={aman.dets.count}')
    fig.tight_layout()
    return fig, ax
    
def plot_noise_vs_opteff(aman):
    obs_id = aman.obs_info.obs_id
    ws = aman.det_info.wafer_slot[0]
    bandpass = aman.det_info.wafer.bandpass[0]
    color = 'tab:blue' if bandpass=='f090' else 'tab:orange'
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(aman.opt_eff * 100, aman.wn_signal * 1e6, '.', alpha=0.3, color=color)
    ax.set_ylim(0, 100, )
    ax.set_xlabel('optical efficiency [%]')
    ax.set_ylabel('white noise level \n[aW/sqrtHz]')
    
    ax.set_title(f'{obs_id}, {ws}, {bandpass}, #={aman.dets.count}')
    fig.tight_layout()
    return fig, ax

def plot_pwv_vs_signal(aman, signal_name='dsT'):
    obs_id = aman.obs_info.obs_id
    ws = aman.det_info.wafer_slot[0]
    bandpass = aman.det_info.wafer.bandpass[0]
    color = 'tab:blue' if bandpass=='f090' else 'tab:orange'
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for di, det in enumerate(aman.dets.vals[:]):
        ax.plot(aman.pwv_class, aman[signal_name][di], color=color, alpha=0.01)
    ax.set_xlabel('pwv [mm]')
    ax.set_ylabel('dsT [pW]')

    ax.set_title(f'{obs_id}, {ws}, {bandpass}, #={aman.dets.count}')
    fig.tight_layout()
    return fig, ax