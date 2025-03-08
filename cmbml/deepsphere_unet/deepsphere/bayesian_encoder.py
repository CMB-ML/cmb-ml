from torch import nn

from .bayesian_utils import BayesianSphericalChebBN2, BayesianSphericalChebBNPool2
    
class Encoder(nn.Module):
    """Encoder for the Spherical UNet.
    """

    def __init__(self, pooling, laps, kernel_size, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        """Initialization.

        Args:
            pooling (:obj:`torch.nn.Module`): pooling layer.
            laps (list): List of laplacians.
            kernel_size (int): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.enc_l3 = BayesianSphericalChebBN2(9, 32, 64, laps[3], self.kernel_size, weight_regularizer, dropout_regularizer)
        self.enc_l2 = BayesianSphericalChebBNPool2(64, 128, 128, laps[2], self.pooling, self.kernel_size, weight_regularizer, dropout_regularizer)
        self.enc_l1 = BayesianSphericalChebBNPool2(128, 256, 256, laps[1], self.pooling, self.kernel_size, weight_regularizer, dropout_regularizer)
        self.enc_l0 = BayesianSphericalChebBNPool2(256, 512, 256, laps[0], self.pooling, self.kernel_size, weight_regularizer, dropout_regularizer)
        

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