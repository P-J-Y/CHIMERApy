import numpy as np
from matplotlib import pyplot as plt


def psf_gaussian(npix, fwhm):
    """
    Generate a PSF with a Gaussian shape.

    Parameters
    ----------
    npix : int
        Number of pixels in the PSF.
    fwhm : float
        FWHM of the Gaussian in pixels.

    Returns
    -------
    psf : numpy.ndarray
        PSF with shape (npix, npix).
    """
    psf = np.zeros((npix, npix))
    x = np.arange(npix)
    y = np.arange(npix)
    x, y = np.meshgrid(x, y)
    psf = np.exp(-0.5 * ((x - npix / 2) ** 2 + (y - npix / 2) ** 2) / fwhm ** 2)
    return psf

psf = psf_gaussian(500, 100)
plt.imshow(psf)
plt.show()