import numpy as np
import sys

def raster_standardize(data, data_type, raster_min, raster_max):
    '''

    :param data: input data to standardize
    :param data_type: data type flag of the input data
    :param raster_min: minimum pixel value needed for min/max standardization
    :param raster_max: maximum pixel value needed for min/max standardization
    :return: Standardized version of input data
    '''
    # standardizing Sentinel MSI layers
    if data_type == "sentinel_msi":
        data[np.isnan(data)] = 0.0
        data[data < 0.0] = 0
        data[data > 10000.0] = 10000.0

        data = data / 10000.0

    # standardizing Sentinel SAR layers TODO: to be added later
    elif data_type == "sentinel_sar":
        sys.exit("Not supporting Sentinel SAR images yet.")

    # standardizing speed related layers: 200km/h as treshold
    elif data_type == "speed":
        data[data < 0.0] = 0.0
        data[data > 200.0] = 200
        data[np.isnan(data)] = 0.0

        data = data / 200.0

    # standardizing bearing layers TODO
    elif data_type == "bearing":
        sys.exit("Not supporting bearing images yet.")

    # standardizing min/max layers
    elif data_type == "min_max":
        data = (data - raster_min) / (raster_max - raster_min)

    # exit if data type is not valid
    else:
        sys.exit(data_type + " is not supported.")

    return data