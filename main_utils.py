import numpy as np
from matplotlib import pyplot as plt
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.time import parse_time
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import Helioprojective

def bytscl(data, top=255, bottom=0, nan_val=0, Max=None, Min=None):
    '''
    Purpose:
        Scale the data to the range [bottom, top]
        If the data is a masked array, the mask will be retained.
        If the data is a masked array, the mask will be retained.
        If the data is a masked array, the mask will be retained.

    Parameters:
        [data] data to be scaled
        [top] the top value of the scaled data
        [bottom] the bottom value of the scaled data
        [nan_val] the value to be assigned to NaN
        [max] the maximum value of the data
        [min] the minimum value of the data

    Example:
        Scale the data to the range [0, 255]
        >> bytscl(data, top=255, bottom=0)

    Written by Gao Yuhang, Mar. 2, 2022
    '''
    if Max is None: Max = np.nanmax(data)
    if Min is None: Min = np.nanmin(data)
    if np.isnan(Max): Max = top
    if np.isnan(Min): Min = bottom
    if Max == Min:
        data = np.ones(data.shape) * top
    else:
        data = (top - bottom) * (data - Min) / (Max - Min) + bottom
    data[data > top] = top
    data[data < bottom] = bottom
    data[np.isnan(data)] = nan_val
    return data


def psf_gaussian(npixel, fwhm):
    '''
    Based on psf_gaussian in Solar Software
    Purpose:
        Create a 2-d Gaussian with specified FWHM
        Return a point spread function having Gaussian profiles as a 2D image
            namely, the 2d gaussian for magnetic cut offs

    Parameters:
        [npixel] number pixels for each dimension; just one number (all sizes equal).
        [fwhm] the desired Full-Width Half-Max (pixels) in each dimension, specify as an array.

    Example:
        Create a 31 x 31 array containing a normalized centered Gaussian
        with an X FWHM = 4.3 and a Y FWHM = 3.6
        >> array = psf_gaussian(31, [4.3,3.6])

    Written by Gao Yuhang, Mar. 2, 2022
    '''
    import numpy as np
    st_dev = fwhm/(2.0 * np.sqrt(2.0 * np.log(2.0)))
    npix = [npixel, npixel]
    ndim = 2
    cntrd = (npixel-1)/2.

    psf = np.zeros(npix)
    x = np.arange(float(npix[0]))-cntrd
    y = np.arange(float(npix[1]))-cntrd

    # psfx = (1/(np.sqrt(2 * np.pi)*st_dev[0]))*np.exp(-x**2/(2*(st_dev[0]**2)))
    # psfy = (1/(np.sqrt(2 * np.pi)*st_dev[1]))*np.exp(-y**2/(2*(st_dev[1]**2)))

    psfx = np.exp(-x**2/(2*(st_dev[0]**2)))
    psfy = np.exp(-y**2/(2*(st_dev[0]**2)))


    for j in range(npix[1]): psf[:,j] = psfx * psfy[j]

    # plt.imshow(psf)

    return psf

def poly_area(x, y):
    '''
    暂时没用这个
    Purpose:
        Calculate the area of a polygon
        The polygon is defined by the coordinates of its vertices.

    Parameters:
        [x] x coordinates of the vertices of the polygon
        [y] y coordinates of the vertices of the polygon

    Example:
        Calculate the area of a polygon
        >> poly_area(x, y)

    Written by Gao Yuhang, Mar. 2, 2022
    '''
    import numpy as np
    xy = np.column_stack((x, y))
    area = 0.5 * np.abs(np.dot(xy[:-1], xy[1:]) - np.dot(xy[1:], xy[:-1]))
    return area

def rot_contour(contour2arc,time_obs,time_last):
    st = parse_time(time_obs)
    et = parse_time(time_last)
    contour2arc_roted = np.zeros(contour2arc.shape)
    c = SkyCoord(contour2arc[:,0] * u.arcsec, contour2arc[:,1] * u.arcsec,
                 obstime=st, observer="earth", frame=Helioprojective)
    c_roted = solar_rotate_coordinate(c, time=et)
    contour2arc_roted[:, 0] = c_roted.Tx.arcsec
    contour2arc_roted[:, 1] = c_roted.Ty.arcsec
    return contour2arc_roted


if __name__ == "__main__":
    psf = psf_gaussian(4096, [2000, 2000])
    print('test')
