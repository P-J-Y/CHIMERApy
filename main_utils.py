import numpy as np


def bytscl(data, top=255, bottom=0, nan_val=0):
    """
    Scale data to range [bottom, top]
    """
    if nan_val is not None:
        data = data.copy()
        data[np.isnan(data)] = nan_val
    data = (data - data.min()) * (top - bottom) / (data.max() - data.min()) + bottom
    return data