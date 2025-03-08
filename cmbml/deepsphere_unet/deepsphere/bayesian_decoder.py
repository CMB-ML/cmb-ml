import torch
import torch.nn as nn

from .bayesian_utils import BayesianSphericalChebBN2

from .dropout import ConcreteDropout

class BayesianSphericalChebBNPoolConcat(nn.Module):
    """Building Block calling a SphericalChebBNPool Block
    then concatenating the output with another tensor
    and calling a SphericalChebBN block.
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size, weight_regularizer=1e-6, dropout_regularizer=1e-5):
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
        self.spherical_cheb_bn = BayesianSphericalChebBN2(in_channels, middle_channels, out_channels, lap, kernel_size, weight_regularizer, dropout_regularizer)

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

    def __init__(self, unpooling, laps, kernel_size, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        """Initialization.

        Args:
            unpooling (:obj:`torch.nn.Module`): The unpooling object.
            laps (list): List of laplacians.
        """
        super().__init__()
        self.unpooling = unpooling
        self.kernel_size = kernel_size
        self.dec_l1 = BayesianSphericalChebBNPoolConcat(256+256, 256, 128, laps[1], self.unpooling, self.kernel_size, weight_regularizer, dropout_regularizer)
        self.dec_l2 = BayesianSphericalChebBNPoolConcat(128 + 128, 128, 64, laps[2], self.unpooling, self.kernel_size, weight_regularizer, dropout_regularizer)
        self.dec_l3 = BayesianSphericalChebBNPoolConcat(64 + 64, 32, 16, laps[3], self.unpooling, self.kernel_size, weight_regularizer, dropout_regularizer)
        # self.dec_fin = SphericalChebConv(1, 1, laps[3], self.kernel_size)
        self.dec_fin_mu = torch.nn.Conv1d(16, 1, 1)
        self.dec_fin_logvar = torch.nn.Conv1d(16, 1, 1)
        self.dec_fin_logvar.weight.data.normal_(0, 1e-6)
        self.dec_fin_logvar.bias.data.fill_(0.01)
        self.cd_mu = ConcreteDropout(channels_first=True, weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.cd_logvar = ConcreteDropout(channels_first=True, weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
    

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
        x = x.permute(0, 2, 1)
        mu = self.cd_mu(x, self.dec_fin_mu)
        logvar = self.cd_logvar(x, self.dec_fin_logvar)
        
        return mu, logvar
