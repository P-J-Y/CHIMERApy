import os

import matplotlib.pyplot as plt

import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, SqrtStretch

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

res = Fido.search(a.Time('2022-01-01T00:00:00', '2022-01-01T00:01:00'),
                  a.Instrument.aia,
                  a.Wavelength(171*u.AA)|a.Wavelength(193*u.AA)|a.Wavelength(211*u.AA),
                  a.Physobs.intensity)

download_files = Fido.fetch(res,path="/Users/gyh/Desktop/research/CH_detect/py/data")

res = Fido.search(a.Time('2022-01-01T00:00:00', '2022-01-01T00:01:00'),
                  a.Instrument.hmi,
                  a.Physobs.los_magnetic_field)
download_files = Fido.fetch(res,path="/Users/gyh/Desktop/research/CH_detect/py/data")

