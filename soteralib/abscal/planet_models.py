import numpy as np
from sotodlib import core, coords
from astropy.constants import au
from soteralib.utils import Tconverter

# Calculate solid angle of planet.
# Now, I assumes a planet as a sphere.
def get_naive_solid_angle(radius, distance):
    solid_angle = np.pi * radius**2 / distance**2
    return solid_angle

# the values comes from NASA JPl (https://ssd.jpl.nasa.gov/planets/phys_par.html)
# Now, the values are just equatorial radius.
planet_radius_dict = {'Mercury': 2440.53e3,
                      'Venus': 6051.8e3,
                      'Mars': 3396.1e3,
                      'Jupiter': 71492e3,
                      'Saturn': 60268e3,
                      'Uranus': 25559e3,
                      'Neptune': 24764e3}

#Thermodynamic temperature of planets.
#Values come from Planck intermediate results LII. Planet flux densities (2017)
#(https://www.aanda.org/articles/aa/abs/2017/11/aa30311-16/aa30311-16.html)
planck_bands = np.array([28.4, 44.1, 70.4, 100, 143, 217, 353, 545, 857]) * 1e9
planck_Tth_Mars = np.array([np.nan, np.nan, np.nan, 194.3, 198.4, 201.9, 209.9, 209.2, 213.6])
planck_Tth_Jupiter = np.array([146.6, 160.9, 173.3, 172.6, 174.1, 175.8, 167.4, 137.4, 161.3])
planck_Tth_Saturn = np.array([138.9, 147.3, 150.6, 145.7, 147.1, 145.1, 141.6, 102.5, 115.6])
planck_Tth_Uranus = np.array([np.nan, np.nan, np.nan, 120.5, 108.4, 98.5, 86.2, 73.9, 66.2])
planck_Tth_Neptune = np.array([np.nan, np.nan, np.nan, 117.4, 106.4, 97.4, 82.6, 72.3, 65.3])

planck2017_Tth_dict =  {'band_freqs': planck_bands,
                        'Mars': planck_Tth_Mars,
                        'Jupiter': planck_Tth_Jupiter,
                        'Saturn': planck_Tth_Saturn,
                        'Uranus': planck_Tth_Uranus,
                        'Neptune': planck_Tth_Neptune}

planck2017_Trj_dict =  {'band_freqs': planck_bands,
                        'Mars': Tconverter.Tth2Trj(planck_Tth_Mars, planck_bands),
                        'Jupiter': Tconverter.Tth2Trj(planck_Tth_Jupiter, planck_bands),
                        'Saturn': Tconverter.Tth2Trj(planck_Tth_Saturn, planck_bands),
                        'Uranus': Tconverter.Tth2Trj(planck_Tth_Uranus, planck_bands),
                        'Neptune': Tconverter.Tth2Trj(planck_Tth_Neptune, planck_bands)}

class PlanetModel:
    """
    This is the class for planet information.
    It is used to store the information about the planet.
    model has the following attributes:
    * source_name: name of the planet
    * freq_band: frequency band for the planet
    * radius: mean radius of the planet
    * timestamp: universal time (utc) for calculation of the planet coordinate
    
    * ra: right ascension of the planet at the timestamp
    * dec: declination of the planet at the timestamp
    * distance: distance between the observer and the planet at the timestamp
    * solid_angle: solid angle of the planet at the timestamp
    * Tth: thermodynamic temperature of the planet in the nearest planck band
    * Trj: RJ temperature of the plaet in the nearest planck band
    """
    def __init__(self, source_name, freq_band, timestamp):
        self.source_name = source_name
        self.freq_band = freq_band
        self.timestamp = timestamp

        ra, dec, distance_au = coords.planets.get_source_pos(source_name, timestamp)
        distance = distance_au * au.si.value
        self.ra = ra
        self.dec = dec
        self.distance = distance
        
        self.radius = planet_radius_dict[source_name]
        self.solid_angle = get_naive_solid_angle(self.radius, self.distance)

        nearest_planck_band_idx = np.argmin(np.abs(planck2017_Tth_dict['band_freqs'] - freq_band))
        self.nearest_planck_band = nearest_planck_band_idx
        self.Tth = planck2017_Tth_dict[source_name][nearest_planck_band_idx]
        self.Trj = planck2017_Trj_dict[source_name][nearest_planck_band_idx]