import numpy as np
import cmbnncs.unet as unet
from src.utils.suppress_print import SuppressPrint


def make_unet(model_dict, max_filters, unet_to_make):
    log_max_filters = int(np.log2(max_filters))

    if unet_to_make == "unet5":
        log_channels_min = log_max_filters - 4
        unet_class = unet.UNet5
    elif unet_to_make == "unet8":
        log_channels_min = log_max_filters - 7
        unet_class = unet.UNet8

    channels = tuple([2**i for i in range(log_channels_min, log_max_filters+1)])

    with SuppressPrint():
        net = unet_class(channels, **model_dict)
    return net
