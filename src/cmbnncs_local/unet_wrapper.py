import numpy as np
import cmbnncs.unet as unet
from ..utils.suppress_print import SuppressPrint


def make_unet(cfg, nside=None):
    if nside is None:
        nside = cfg.experiment.nside
    max_filters = cfg.model.max_filters
    input_channels = cfg.experiment.detector_freqs
    kernels_size = cfg.model.kernels_size
    strides = cfg.model.strides
    mainActive = cfg.model.mainActive
    finalActive = cfg.model.finalActive
    finalBN = cfg.model.finalBN

    log_max_filters = int(np.log2(max_filters))

    if cfg.model.unet_to_make == "unet5":
        log_channels_min = log_max_filters - 4
        unet_class = unet.UNet5
    elif cfg.model.unet_to_make == "unet8":
        log_channels_min = log_max_filters - 7
        unet_class = unet.UNet8

    input_c = len(input_channels)
    sides = (nside ,nside)
    channels = tuple([2**i for i in range(log_channels_min, log_max_filters+1)])

    with SuppressPrint():
        net = unet_class(channels,
                         channel_in=input_c,
                         channel_out=1,
                         kernels_size=kernels_size,
                         strides=strides,
                         extra_pads=None,
                         mainActive=mainActive,
                         finalActive=finalActive,
                         finalBN=finalBN, 
                         sides=sides)
    return net


# def make_unet8(cfg, nside=None):
#     if nside is None:
#         nside = cfg.experiment.nside
#     max_filters = cfg.model.max_filters
#     input_channels = cfg.experiment.detector_freqs
#     kernels_size = cfg.model.kernels_size
#     strides = cfg.model.strides
#     mainActive = cfg.model.mainActive
#     finalActive = cfg.model.finalActive
#     finalBN = cfg.model.finalBN

#     log_max_filters = int(np.log2(max_filters))
#     channels = tuple([2**i for i in range(log_max_filters-7, log_max_filters+1)])
#     input_c = len(input_channels)
#     sides = (nside ,nside)

#     # with SuppressPrint():
#     #     net = unet.UNet5(channels,
#     #                     channel_in=input_c,
#     #                     channel_out=1,
#     #                     kernels_size=kernels_size,
#     #                     strides=strides,
#     #                     extra_pads=None,
#     #                     mainActive=mainActive,
#     #                     finalActive=finalActive,
#     #                     finalBN=finalBN, 
#     #                     sides=sides)
#     # return net
