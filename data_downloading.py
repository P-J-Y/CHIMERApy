import os

import matplotlib.pyplot as plt

import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, SqrtStretch

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import shutil

datapath = os.getcwd()+'\\data'
# filelt = os.listdir(datapath)
# if len(filelt)>0:
#     shutil.rmtree(datapath)
#     os.mkdir(datapath)

t1 = '2013-03-27T00:00:00'
t2 = '2013-03-27T00:02:00'

res = Fido.search(a.Time(t1, t2),
                  a.Instrument.aia,
                  a.Wavelength(171*u.AA)|a.Wavelength(193*u.AA)|a.Wavelength(211*u.AA),
                  a.Physobs.intensity,
                  a.Sample(3*u.minute))

download_files = Fido.fetch(res,path=datapath)

res = Fido.search(a.Time(t1, t2),
                  a.Instrument.hmi,
                  a.Physobs.los_magnetic_field,
                  a.Sample(3*u.minute))
download_files = Fido.fetch(res,path=datapath)

