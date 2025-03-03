from torch import nn

from .utils import SphericalChebBN2, SphericalChebBNPool2
    
class Encoder(nn.Module):
    """Encoder for the Spherical UNet.
    """

    def __init__(self, pooling, laps, kernel_size):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            kernel_size (int): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.enc_l3 = SphericalChebBN2(9, 32, 64, laps[3], self.kernel_size)
        self.enc_l2 = SphericalChebBNPool2(64, 128, 128, laps[2], self.pooling, self.kernel_size)
        self.enc_l1 = SphericalChebBNPool2(128, 256, 256, laps[1], self.pooling, self.kernel_size)
        self.enc_l0 = SphericalChebBNPool2(256, 512, 256, laps[0], self.pooling, self.kernel_size)
        

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            x_enc* :obj: `torch.Tensor`: output [batch x vertices x channels/features]
        """
        x_enc3 = self.enc_l3(x)
        x_enc2 = self.enc_l2(x_enc3)
        x_enc1 = self.enc_l1(x_enc2)
        x_enc0 = self.enc_l0(x_enc1)

        return x_enc0, x_enc1, x_enc2, x_enc3