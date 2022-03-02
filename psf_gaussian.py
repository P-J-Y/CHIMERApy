def paf_gaussian(npixel, fwhm):
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

    psfx = (1/(np.sqrt(2 * np.pi)*st_dev[0]))*np.exp(-x**2/(2*(st_dev[0]**2)))

    psfy = (1/(np.sqrt(2 * np.pi)*st_dev[1]))*np.exp(-y**2/(2*(st_dev[1]**2)))

    for j in range(npix[1]): psf[:,j] = psfx * psfy[j]

    # plt.imshow(psf)

    return

