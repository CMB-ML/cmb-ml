"""
Written using ChatGPT with instructions to produce the simplest implementation of a UNet model,
with parameterized depth (for simplicity when debugging: a very shallow UNet can be produced).

Some modifications were made.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_features, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_features, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, kernel_size, padding)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        out_channels = in_channels // 2
        self.up = nn.ConvTranspose2d(in_channels,
                                     out_channels,
                                     kernel_size=2,
                                     stride=2)
        self.conv = DoubleConv(in_channels,  # between up and conv, we will cat
                               out_channels,
                               kernel_size,
                               padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SimpleUNetModel(nn.Module):
    def __init__(
                 self,
                 detectors,
                 map_fields,
                 im_size=128,
                 n_init_features=32,
                 n_dns=3,
                 note='',
                 ):
        super().__init__()
        self.note = note

        if map_fields != 'I':
            raise NotImplementedError("This model only supports I maps.")

        self.n_dets = len(detectors)
        self.im_size = im_size                      # 128 Patch N_side (outside this model)

        i_init_feat = int(np.log2(n_init_features)) # 32 -> 5
        i_cent_feat  = i_init_feat + n_dns          # 5 + 3 = 8

        # Define chunks of layers
        self.in_c = DoubleConv(in_features=self.n_dets,
                               out_channels=n_init_features,
                               kernel_size=(3, 3),
                               padding=(1, 1)
                               )

        self.dns = nn.ModuleList(
            [Down(in_channels=2**i,
                  ) for i in range(i_init_feat, i_cent_feat)]
        )

        self.ups = nn.ModuleList(
            [Up(in_channels=2**i,
                ) for i in range(i_cent_feat, i_init_feat, -1)]
        )

        self.out = OutConv(in_channels=n_init_features, out_channels=1)

    def forward(self, x):
        x = self.in_c(x)
        downs = [x]
        for i, down in enumerate(self.dns):
            x = down(x)
            downs.append(x)
        for i, up in enumerate(self.ups):
            x = up(x, downs[-(i+2)])
        x = self.out(x)
        return x
