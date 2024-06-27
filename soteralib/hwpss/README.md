This module is for analyzing HWPSS and Instrumental Polarization for SO SAT

# Naming convention
Time Ordered data
* `aman.signal`
  * raw tod or pre-processed data
* `aman.dsT`, `aman.demodQ` and `aman.demodU`
  * The low-pass filtered/demodulated data
* `aman.hwpss_model`
  * Modeled TOD of HWPSS by the hwpss binning code
* `aman.pwv_class`
  * pwv measured by radiometer around class
  

HWPSS Statistics
* `aman.hwpss_stats`
  * binned_angle
  * binned_signal
  * sigma_bin
  * binned_model
  * coeffs
  * covars
  * redchi2s

Leakage Statistics
(https://github.com/simonsobs/sotodlib/blob/preproc_sat_mapmake/sotodlib/tod_ops/t2pleakage.py)
* `aman.leakage_stats`
  * `coeffsQ` and `coeffsU`
  * `sigma_coeffsQ` and `sigma_coeffsU`





# TOD tools
