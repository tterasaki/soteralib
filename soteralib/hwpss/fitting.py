import numpy as np
from iminuit import Minuit

# define the model function
def model_uniform(theta_fp, phi_fp, uniform_P, uniform_theta):
    Q_uniform = uniform_P * np.cos(2*uniform_theta) * np.ones(theta_fp.shape[0])
    U_uniform = uniform_P * np.sin(2*uniform_theta) * np.ones(theta_fp.shape[0])
    return np.array([Q_uniform, U_uniform])

def model_radial(theta_fp, phi_fp, radial_P_pivot, radial_alpha):
    theta_fp_pivot = np.deg2rad(17.5)
    
    radial_P = radial_P_pivot * (theta_fp/theta_fp_pivot)**radial_alpha
    Q_radial = radial_P * np.cos(2 * phi_fp)
    U_radial = radial_P * np.sin(2 * phi_fp)
    return np.array([Q_radial, U_radial])
    
    
def model(theta_fp, phi_fp, 
           uniform_P, uniform_theta, 
           radial_P_pivot, radial_alpha):
    
    Q_uniform, U_uniform = model_uniform(theta_fp, phi_fp, uniform_P, uniform_theta)
    Q_radial, U_radial = model_radial(theta_fp, phi_fp, radial_P_pivot, radial_alpha)
    
    Q = Q_uniform + Q_radial
    U = U_uniform + U_radial
    return np.array([Q, U])

def do_fit(aman, mode='hwpss4f', err_relax_factor=1):
    Qval_name = f'{mode}_Q_tele_val'
    Qerr_name = f'{mode}_Q_tele_err'
    Uval_name = f'{mode}_U_tele_val'
    Uerr_name = f'{mode}_U_tele_err'
    thetaval_name = f'{mode}_theta_tele_err'
    
    # generate some sample data with errors
    theta_fp, phi_fp = aman.focal_plane.theta_fp, aman.focal_plane.phi_fp
    
    Z = np.array([aman[Qval_name], aman[Uval_name]])
    Z_err = np.array([aman[Qerr_name], aman[Uerr_name]]) * err_relax_factor

    # define the chi-squared function
    def chi2(uniform_P, uniform_theta, radial_P_pivot, radial_alpha):
        expected = model(theta_fp, phi_fp, uniform_P, uniform_theta, radial_P_pivot, radial_alpha)
        return np.sum(((Z - expected) / Z_err)**2)

    # create a Minuit object and set the initial parameters
    uniform_P_init = np.mean(aman[Qval_name])**2 + np.mean(aman[Uval_name])**2
    uniform_theta_init = np.mean(aman[thetaval_name])
    radial_P_pivot_init = np.percentile(np.sqrt(aman[Qval_name]**2 + aman[Uval_name]**2), 90)
    radial_alpha_init = 2.0
    
    m = Minuit(chi2, uniform_P=uniform_P_init, uniform_theta=uniform_theta_init, 
               radial_P_pivot=radial_P_pivot_init, radial_alpha=radial_alpha_init)
    
    # m.fixed['uniform_theta'] = True  # Fix uniform_theta
    # m.fixed['uniform_P'] = True  # Uncomment to fix uniform_P
    # m.fixed['radial_P_pivot'] = True  # Uncomment to fix radial_P_pivot
    m.fixed['radial_alpha'] = True  # Uncomment to fix radial_alpha
    
    # fit the model to the data
    m.migrad()
    m.minos()
    
    return m

def wrap_model(aman, m, mode='hwpss4f'):
    phi_fp = aman.focal_plane.phi_fp
    params = m.values.to_dict()
    Q_uniform_model, U_uniform_model = model_uniform(aman.focal_plane.theta_fp, aman.focal_plane.phi_fp, 
                                         params['uniform_P'], params['uniform_theta'])
    Q_radial_model, U_radial_model = model_radial(aman.focal_plane.theta_fp, aman.focal_plane.phi_fp, 
                                         params['radial_P_pivot'], params['radial_alpha'])                                    

    Q_total_model = Q_uniform_model + Q_radial_model
    U_total_model = U_uniform_model + U_radial_model

    P_uniform_model = np.sqrt(Q_uniform_model**2 + U_uniform_model**2)
    P_radial_model = np.sqrt(Q_radial_model**2 + U_radial_model**2)
    P_total_model = np.sqrt(Q_total_model**2 + U_total_model**2)

    theta_uniform_model = 0.5 * np.arctan2(U_uniform_model, Q_uniform_model)
    theta_radial_model = 0.5 * np.arctan2(U_radial_model, Q_radial_model)
    theta_total_model = 0.5 * np.arctan2(U_total_model, Q_total_model)

    P_radialQ_uniform_model = Q_uniform_model * np.cos(2*phi_fp) +  U_uniform_model * np.sin(2*phi_fp)
    P_radialU_uniform_model = -Q_uniform_model * np.sin(2*phi_fp) +  U_uniform_model * np.cos(2*phi_fp)
    P_radialQ_radial_model = Q_radial_model * np.cos(2*phi_fp) +  U_radial_model * np.sin(2*phi_fp)
    P_radialU_radial_model = -Q_radial_model * np.sin(2*phi_fp) +  U_radial_model * np.cos(2*phi_fp)
    P_radialQ_total_model = Q_total_model * np.cos(2*phi_fp) +  U_total_model * np.sin(2*phi_fp)
    P_radialU_total_model = -Q_total_model * np.sin(2*phi_fp) +  U_total_model * np.cos(2*phi_fp)

    uniform_model_param_names = [f'{mode}_Q_uniform_model', f'{mode}_U_uniform_model', 
                             f'{mode}_P_uniform_model', f'{mode}_theta_uniform_model',
                             f'{mode}_P_radialQ_uniform_model', f'{mode}_P_radialU_uniform_model']

    uniform_model_params = [Q_uniform_model, U_uniform_model, P_uniform_model, theta_uniform_model,
                                 P_radialQ_uniform_model, P_radialU_uniform_model]

    radial_model_param_names = [f'{mode}_Q_radial_model', f'{mode}_U_radial_model', 
                             f'{mode}_P_radial_model', f'{mode}_theta_radial_model',
                             f'{mode}_P_radialQ_radial_model', f'{mode}_P_radialU_radial_model']

    radial_model_params = [Q_radial_model, U_radial_model, P_radial_model, theta_radial_model,
                                 P_radialQ_radial_model, P_radialU_radial_model]

    total_model_param_names = [f'{mode}_Q_total_model', f'{mode}_U_total_model', 
                             f'{mode}_P_total_model', f'{mode}_theta_total_model',
                             f'{mode}_P_radialQ_total_model', f'{mode}_P_radialU_total_model']

    total_model_params = [Q_total_model, U_total_model, P_total_model, theta_total_model,
                                 P_radialQ_total_model, P_radialU_total_model]
    
    for name, param in zip(uniform_model_param_names, uniform_model_params):
        aman.wrap(name, param)
    for name, param in zip(radial_model_param_names, radial_model_params):
        aman.wrap(name, param)
    for name, param in zip(total_model_param_names, total_model_params):
        aman.wrap(name, param)
        
    for key, val in m.values.to_dict().items():
        aman.wrap(f'{mode}_{key}_val', val)
    for key, val in m.errors.to_dict().items():
        aman.wrap(f'{mode}_{key}_err', val)
    redchi2 = m.fval / (aman.dets.count*2 - len(m.values) + np.count_nonzero(m.fixed.to_dict().values()))
    aman.wrap(f'{mode}_redchi2', redchi2)
        
    return

def do_main(aman):
    modes = ['hwpss4f', 'leakage4f']
    for mode in modes:
        m = do_fit(aman, mode=mode, err_relax_factor=1)
        wrap_model(aman, m, mode=mode)
    return



# aman = core.AxisManager.load('result01_combined/obs_1714176018_satp1_1111111_f150.hdf')
# modes = ['hwpss4f', 'leakage4f']
# for mode in modes:
#     m = do_fit(aman, mode=mode, err_relax_factor=1)
#     wrap_model(aman, m, mode=mode)
# ah_plot.plot_quiver(aman, P_name='hwpss4f_P_tele_val', theta_name='hwpss4f_theta_tele_val')
# ah_plot.plot_quiver(aman, P_name='hwpss4f_P_total_model', theta_name='hwpss4f_theta_total_model')
# ah_plot.plot_quiver(aman, P_name='leakage4f_P_tele_val', theta_name='leakage4f_theta_tele_val')
# ah_plot.plot_quiver(aman, P_name='leakage4f_P_total_model', theta_name='leakage4f_theta_total_model')
