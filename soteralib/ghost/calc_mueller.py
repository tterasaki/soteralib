import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import transfer_matrix as tm

########## AR parameters #############
# indexes
sapphire_no = 3.05
sapphire_ne = 3.38
alumina_n = 3.14
mullite_n = 2.52
duroid_n = 1.41
epo_n = 1.7
# loss tangents
sapphire_tan = 1e-4 #ar.sapphire_tan_100_MF
alumina_tan = 3.7e-4 #ar.alumina_tan_100_MF
mullite_tan = 121e-4 #ar.mullite_tan_100_MF
duroid_tan = 12e-3 #ar.duroid_tan_100_MF
epo_tan = 0.0
# thicknesses
d_sapphire=3.75
d_alumina =3.0
d_mullite =0.212
d_duroid = 0.394
d_epo = 0.04
# theta2 for AHWP
theta2=54.

########## Material objects #############
sapphire = tm.material( sapphire_no, sapphire_ne, sapphire_tan, sapphire_tan, 'Sapphire', materialType='uniaxial')
alumina  = tm.material( alumina_n, alumina_n, alumina_tan, alumina_tan, 'Alumina', materialType='isotropic')
mullite  = tm.material( mullite_n, mullite_n, mullite_tan, mullite_tan, 'Mullite', materialType='isotropic')
duroid   = tm.material( duroid_n, duroid_n, duroid_tan, duroid_tan, 'Duroid', materialType='isotropic')
epo = tm.material(epo_n, epo_n, epo_tan, epo_tan, 'Epo-tek', materialType='isotropic')

########## IR filter #################
thicknesses_ir = np.array([d_duroid, d_epo, d_mullite, 
                            d_alumina, 
                            d_mullite, d_epo, d_duroid])*tm.mm
materials_ir   = [duroid, epo, mullite, 
                   alumina,
                   mullite, epo, duroid]
angles_ir      = [0., 0., 0., 
                   0., 
                   0., 0., 0.]
ir_filter = tm.Stack( thicknesses_ir, materials_ir, angles_ir )

############### AHWP ####################
thicknesses_hwp = np.array([d_duroid, d_epo, d_mullite, d_alumina, 
                        d_sapphire, d_sapphire, d_sapphire,
                        d_alumina, d_mullite, d_epo, d_duroid])*tm.mm
materials_hwp   = [duroid, epo, mullite, alumina,
               sapphire, sapphire, sapphire,
               alumina, mullite, epo, duroid]
angles_hwp      = [0., 0., 0., 0., 
               0., np.radians(theta2), 0.,
               0., 0., 0., 0.]
hwp         = tm.Stack( thicknesses_hwp, materials_hwp, angles_hwp )


###### Example of getting Mueller matrix ######
#M = tm.Mueller(ir_filter, freq*tm.GHz, incidenceAngle=np.deg2rad(17), 
#                rotation=np.deg2rad(0.), reflected=True)
