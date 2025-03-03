import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .laplacian import get_laplacians

from .pooling import Pool

class SphericalUNet(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, N, depth, laplacian_type, kernel_size):
        """Initialization.

        Args:
            N (int): Number of pixels in the input image
            depth (int): The depth of the UNet, which is bounded by the N and the type of pooling
            kernel_size (int): chebychev polynomial degree
        """
        super().__init__()
        self.kernel_size = kernel_size
        
        self.pooling_class = Pool()
        self.laps = get_laplacians(N, depth, laplacian_type)
    
        self.encoder = Encoder(self.pooling_class.pool, self.laps, self.kernel_size)
        self.decoder = Decoder(self.pooling_class.upsample, self.laps, self.kernel_size)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        x_encoder = self.encoder(x)
        output = self.decoder(*x_encoder)
        return output