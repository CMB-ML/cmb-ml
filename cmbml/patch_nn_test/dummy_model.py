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
    def __init__(self, in_features, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_features, out_channels, 
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.up = nn.ConvTranspose2d(in_channels,
                                     out_channels,
                                     kernel_size=2,
                                     stride=2)
        # between up and conv, we will concatenate feature maps,
        #   doubling the size. So we use in_channels here:
        self.conv = DoubleConv(in_channels,
                               out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
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
                 n_in_channels,  # Should be the number of detectors
                 n_init_features=32,
                 n_dns=3,
                 note='',
                 ):
        super().__init__()
        self.note = note

        self.in_c = DoubleConv(in_features=n_in_channels,
                               out_channels=n_init_features)

        downs = []
        curr_features = n_init_features
        for _ in range(n_dns):
            downs.append(Down(curr_features))
            curr_features *= 2
        self.dns = nn.ModuleList(downs)

        ups = []
        for _ in range(n_dns):
            ups.append(Up(curr_features))
            curr_features //= 2
        self.ups = nn.ModuleList(ups)

        self.out = OutConv(in_channels=n_init_features, out_channels=1)

    def forward(self, x):
        skips = []
        x = self.in_c(x)
        # Top of the UNet
        skips.append(x)

        for down in self.dns:
            x = down(x)
            skips.append(x)

        for i, up in enumerate(self.ups):
            # Pair skip layers. [-(i+2)] is the i-th layer from the end.
            #   It's worth drawing this out by hand to verify...
            x = up(x, skips[-(i+2)])

        x = self.out(x)
        return x
