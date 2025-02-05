"""
Hard-coded implementation of Casas' CENN
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size)
            nn.LeakyReLU(inplace=True)
            nn.MaxPool2d(kernel_size=2)
        )
    
    def forward(self, x):
        return self.conv(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
            nn.LeakyReLU(inplace=True)
            nn.MaxPool2d(kernel_size=2)
        )

    # Concatenate with skip layer
    def forward(self, x1, x2):
        return self.deconv(torch.cat((x1, x2), dim=1))

    # For last deconv that doesn't have a skip layer
    def forward(self, x):
        return self.deconv(x)

class CENN(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_channels = [3, 8, 16, 64, 128, 256]
        out_channels = [8, 16, 64, 128, 256, 512]
        kernel_sizes = [9, 9, 7, 7, 5, 3]

        self.convs = nn.ModuleList(
            [ConvBlock(in_channels[i], out_channels[i], kernel_sizes[i]) for i in range(6)]
        )

        in_channels = [512, 256, 128, 64, 16, 8]
        out_channels = [256, 128, 64, 16, 8, 1]
        kernel_sizes = [3, 5, 7, 7, 9, 9]

        self.deconvs = nn.ModuleList(
            [DeconvBlock(in_channels[i], out_channels[i], kernel_sizes[i]) for i in range(6)]
        )

    def forward(self, x):
        skips = []

        for i, conv in enumerate(self.convs):
            x = conv(x)
            skips.append(x)

        for i, deconv in enumerate(self.deconvs):
            if i < 5:
                x = deconv(x, skips[4-i])
            else:
                x = deconv(x)

        return x