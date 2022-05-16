import os

import matplotlib.pyplot as plt

import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, SqrtStretch

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

res = Fido.search(a.Time('2013-03-01T00:00:00', '2013-03-01T00:10:00'),
                  a.Instrument.aia,
                  a.Wavelength(171*u.AA)|a.Wavelength(193*u.AA)|a.Wavelength(211*u.AA),
                  a.Physobs.intensity,
                  a.Sample(3*u.minute))

download_files = Fido.fetch(res,path=os.getcwd()+'\\data')

res = Fido.search(a.Time('2013-03-01T00:00:00', '2013-03-01T00:10:00'),
                  a.Instrument.hmi,
                  a.Physobs.los_magnetic_field,
                  a.Sample(3*u.minute))
download_files = Fido.fetch(res,path=os.getcwd()+'\\data')

