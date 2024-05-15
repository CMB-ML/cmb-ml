"""
MIT License

Copyright (c) 2020 Laure Vancauwenberghe, Michael Allemann, Yoann Ponti, Basile Chatillon, Lionel Martin, Michaël Defferrard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Code derived from https://github.com/deepsphere/deepsphere-pytorch
# More specifically, https://github.com/deepsphere/deepsphere-pytorch/blob/master/deepsphere/models/spherical_unet/unet_model.py

from typing import Union, List

import numpy as np

import torch

import healpy as hp

from .laplacian import healpix_laplacian
from .pooling import HealpixPooling, HealpixMaxPool, HealpixAvgPool, HealpixMaxUnpool, HealpixAvgUnpool
from .conv import SphericalChebConv
from .layer_linear import LinearCombLayer


class PetroffNet(torch.nn.Module):
    """
    An implementation of Petroff's "Autoencoder."
    """

    def __init__(self,
                 n_dets: int,
                 nside: int,
                 laplacian_type: str="normalized",
                 kernel_size: int=9,
                 n_features: int=6):
        """Initialization.

        Args:
            nside (int): Healpix resolution of input maps
            laplacian_type (str): Either "combinatoric" or "normalized"
            kernel_size (int): chebychev polynomial degree
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.pooling_class = HealpixPooling()
        self.n_dets = n_dets
        # We have a "scaling" level for each downscale/upscale; halve the nside until it is 1
        self.n_scalings = np.log2(nside)
        assert int(self.n_scalings) == self.n_scalings, "nside must be a positive power of 2"
        self.n_scalings = int(self.n_scalings)
        powers = np.arange(1, self.n_scalings+1)
        self.nsides = [pow(2, power) for power in powers]
        features = [1, *[n_features] * len(self.nsides)]
        # laps = [healpix_laplacian(this_nside, laplacian_type, dtype=torch.float64) for this_nside in self.nsides]
        laps = [healpix_laplacian(this_nside, laplacian_type) for this_nside in self.nsides]

        self.encoders = torch.nn.ModuleList()
        for _ in range(n_dets):
            self.encoders.append(SingleDetEncoder(self.pooling_class.pooling,
                                                  self.nsides[::-1],
                                                  laps[::-1],
                                                  self.kernel_size,
                                                  features))

        # Major order: detectors/ minor order: nsides
        self.skip_linears = torch.nn.ModuleList([
            torch.nn.ModuleList([LinearCombLayer() for det_n in range(n_dets)])
            for scale_n in self.nsides
        ])
        self.skip_adds = torch.nn.ModuleList([
            AddLayer() for scale_n in self.nsides
        ])

        self.bottleneck = Bottleneck()

        self.decoder = Decoder(self.pooling_class.unpooling,
                               self.nsides,
                               laps,
                               self.kernel_size,
                               features[::-1])

    def forward(self, input_map_data):
        """Forward Pass.

        Args:
            input_map_data (:obj:`torch.Tensor`): B x D x N tensor.
                                                  B: Batch size
                                                  D: Number of detectors
                                                  N: Npix of map

        Returns:
            :obj:`torch.Tensor`: output
        """
        down_skips = []
        down_outs  = []
        # in_skips = [[] for _ in range(len(self.encoders[0].features))]
        
        # Encoder
        # Build up the layers in detector-minor/nsides-major order
        # Iterate over each encoder (per each detector)
        for det_n, encoder in enumerate(self.encoders):
            these_skips, this_out = encoder(input_map_data[:, det_n, :])
            down_skips.append(these_skips)
            down_outs.append(this_out)
        down_out = torch.stack(down_outs)

        # Skips
        up_skips = []
        for scale_n in range(self.n_scalings):
            these_skips = []
            for det_n in range(self.n_dets):
                these_skips.append(self.skip_linears[scale_n][det_n](down_skips[det_n][scale_n]))
            up_skips.append(self.skip_adds[scale_n](
                torch.stack(these_skips)
            ))

        up_skips = list(reversed(up_skips))

        # Bottleneck
        x = self.bottleneck(down_out)

        # Decoder
        x = self.decoder(x, up_skips)
        return x


class SingleDetEncoder(torch.nn.Module):
    def __init__(self, 
                 pooling: Union[HealpixMaxPool, HealpixAvgPool], 
                 nsides, 
                 laps, 
                 kernel_size,
                 features
                 ):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList()

        paired_features = list(zip(features, features[1:]))

        for io_feats, nside, lap in zip(paired_features, nsides, laps):
            in_channels = io_feats[0]
            out_channels = io_feats[1]
            self.encoder_layers.append(Down(in_channels, 
                                            out_channels, 
                                            pooling, 
                                            nside, 
                                            lap, 
                                            kernel_size))

    def forward(self, x):
        skips = []
        x = x.unsqueeze(-1)      # The input needs the single in_channel
        for down_step in self.encoder_layers:
            this_skip, x = down_step(x)
            skips.append(this_skip)
        return skips, x


class Bottleneck(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = AddLayer()
        self.act = torch.nn.ELU(alpha=1)

    def forward(self, x):
        x = self.add(x)
        x = self.act(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self,
                 pooling,
                 nsides,
                 laps,
                 kernel_size,
                 features
                 ):
        super().__init__()
        self.decoder_layers = torch.nn.ModuleList()

        paired_features = list(zip(features, features[1:]))

        for io_feats, nside, lap in zip(paired_features, nsides, laps):
            in_channels = io_feats[0]
            out_channels = io_feats[1]
            self.decoder_layers.append(Up(
                in_channels, 
                out_channels,
                pooling,
                nside,
                lap,
                kernel_size
            ))

    def forward(self, x, skips):
        for up_step, skip in zip(self.decoder_layers, skips):
            x = up_step(x, skip)
        B, N, Fout = x.shape
        assert Fout == 1, "Output of model should have a single ML feature"
        # TODO: Fix:
        x = x.view([B, 1, N])  # Hard-coding a single "field" (e.g. temperature only, no polarization)
        return x


class Down(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 pooling: Union[HealpixMaxPool, HealpixAvgPool], 
                 nside, 
                 lap, 
                 kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # self.pooling = pooling
        self.conv1 = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        # for alpha value: Petroff uses tf.keras.layers.Activation("elu") 
        #                     without specifying alpha.
        #                     Both keras and PyTorch use alpha=1 as default.
        self.act1  = torch.nn.ELU(alpha=1)
        self.conv2 = SphericalChebConv(out_channels, out_channels, lap, kernel_size)
        self.act2  = torch.nn.ELU(alpha=1)
        self.pool  = pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        skip = self.act2(x)
        x = self.pool(skip)
        return skip, x


class Up(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 pooling: Union[HealpixMaxUnpool, HealpixAvgUnpool], 
                 nside, 
                 lap, 
                 kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool  = pooling
        self.conv1 = SphericalChebConv(in_channels, in_channels, lap, kernel_size)
        self.add   = AddLayer()
        self.act1  = torch.nn.ELU(alpha=1)
        self.conv2 = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.act2  = torch.nn.ELU(alpha=1)

    def forward(self, x, skip):
        x = self.pool(x)
        x = self.conv1(x)
        x = torch.stack([x, skip])
        x = self.add(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x

class AddLayer(torch.nn.Module):
    def __init__(self, axis=0):
        super().__init__()
        self.sum_axis = axis

    def forward(self, inputs):
        return torch.sum(inputs, dim=self.sum_axis)
