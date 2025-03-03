import torch
import torch.nn as nn

from .utils import SphericalChebBN2, SphericalChebBNPool
from .chebyshev import SphericalChebConv

class SphericalChebBNPoolConcat(nn.Module):
    """Building Block calling a SphericalChebBNPool Block
    then concatenating the output with another tensor
    and calling a SphericalChebBN block.
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb_bn = SphericalChebBN2(in_channels, middle_channels, out_channels, lap, kernel_size)

    def forward(self, x, concat_data):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]
            concat_data (:obj:`torch.Tensor`): encoder layer output [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        # pylint: disable=E1101
        x = torch.cat((x, concat_data), dim=2)
        # pylint: enable=E1101
        x = self.spherical_cheb_bn(x)
        return x
    
class Decoder(nn.Module):
    """The decoder of the Spherical UNet.
    """

    def __init__(self, unpooling, laps, kernel_size):
        """Initialization.

        Args:
            unpooling (:obj:`torch.nn.Module`): The unpooling object.
            laps (list): List of laplacians.
        """
        super().__init__()
        self.unpooling = unpooling
        self.kernel_size = kernel_size
        self.dec_l1 = SphericalChebBNPoolConcat(256+256, 256, 128, laps[1], self.unpooling, self.kernel_size)
        self.dec_l2 = SphericalChebBNPoolConcat(128 + 128, 128, 64, laps[2], self.unpooling, self.kernel_size)
        self.dec_l3 = SphericalChebBNPoolConcat(64 + 64, 32, 16, laps[3], self.unpooling, self.kernel_size)
        # self.dec_fin = SphericalChebConv(1, 1, laps[3], self.kernel_size)
        self.dec_fin = torch.nn.Conv1d(16, 1, 1)
    

    def forward(self, x_enc0, x_enc1, x_enc2, x_enc3):
        """Forward Pass.

        Args:
            x_enc* (:obj:`torch.Tensor`): input tensors.

        Returns:
            :obj:`torch.Tensor`: output after forward pass.
        """
        x = self.dec_l1(x_enc0, x_enc1)
        x = self.dec_l2(x, x_enc2)
        x = self.dec_l3(x, x_enc3)
        x = self.dec_fin(x.permute(0, 2, 1))

        return x