import lmfit as lf

def polarization_model(x, y, 
                       pol_uniform_P, pol_uniform_theta, 
                       pol_radial_P_pivot, pol_radial_alpha):
    pol_radial_r_pivot = 175
    Q_uniform = pol_uniform_P * np.cos(2*pol_uniform_theta)
    U_uniform = pol_uniform_P * np.sin(2*pol_uniform_theta)
    
    r = np.sqrt(x**2 + y**2)
    pol_radial_P = pol_radial_P_pivot * (r/pol_radial_r_pivot)**pol_radial_alpha
    
    theta = np.arctan2(y, x)
    Q_radial = pol_radial_P * np.cos(2 * theta)
    U_radial = pol_radial_P * np.sin(2 * theta)
    
    Q_add = Q_uniform + Q_radial
    U_add = U_uniform + U_radial
    return Q_add, U_add



def calculate_model(params:lf.Parameters, x:np.ndarray, y:np.ndarray):
    return  polarization_model(x, y,
                             params['pol_uniform_P'], params['pol_uniform_theta'],
                             params['pol_radial_P_pivot'], params['pol_radial_alpha'])

def objective(params:lf.Parameters, x:np.ndarray, y:np.ndarray, 
              Q_data:np.ndarray, U_data:np.ndarray):
    Q_model, U_model = calculate_model(params, x, y) 
    residual = np.sqrt( (Q_data - Q_model)**2 + (U_data - U_model)**2 )
    #residual = (Q_data - Q_model)**2 + (U_data - U_model)**2
    
    return residual

def do_fit(x_data, y_data, Q_data, U_data):
    fit_params = lf.Parameters()
    fit_params.add(name='pol_uniform_P', value=0., min=0.2, max=0.1)
    fit_params.add(name='pol_uniform_theta', value=np.pi/2, min=0., max=2*np.pi)
    fit_params.add(name='pol_radial_P_pivot', value=1.0, min=0.0, max=2.0)
    fit_params.add(name='pol_radial_alpha', value=2.0, min=0.0, max=4.0)
    
    result = lf.minimize(fcn=objective, params=fit_params, method='leastsq', 
                     args=(x_data, y_data, Q_data, U_data))
    return result