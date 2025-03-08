import torch.nn.functional as F
from torch import nn

from .chebyshev import SphericalChebConv
from .bayesian_cheb import ConcreteDropoutChebConv



class BayesianSphericalChebBN(nn.Module):
    """Building Block with a Chebyshev Convolution, Batchnormalization, and ReLu activation.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb = ConcreteDropoutChebConv(in_channels, out_channels, lap, kernel_size, weight_regularizer, dropout_regularizer)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb(x)
        x = self.batchnorm(x.permute(0, 2, 1))
        x = F.relu(x.permute(0, 2, 1))
        return x
    
    
class BayesianSphericalChebBNPool(nn.Module):
    """Building Block with a pooling/unpooling, a calling the SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size, weight_regularizer=1e-6, dropout_regularizer=1e-5):
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
        self.spherical_cheb_bn = BayesianSphericalChebBN(in_channels, out_channels, lap, kernel_size, weight_regularizer, dropout_regularizer)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb_bn(x)
        return x

    
class BayesianSphericalChebBN2(nn.Module):
    """Building Block made of 2 Building Blocks (convolution, batchnorm, activation).
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, kernel_size, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            middle_channels (int): middle number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            kernel_size (int, optional): polynomial degree.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spherical_cheb_bn_1 = BayesianSphericalChebBN(in_channels, middle_channels, lap, kernel_size, weight_regularizer, dropout_regularizer)
        self.spherical_cheb_bn_2 = BayesianSphericalChebBN(middle_channels, out_channels, lap, kernel_size, weight_regularizer, dropout_regularizer)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb_bn_1(x)
        x = self.spherical_cheb_bn_2(x)
        return x

class BayesianSphericalChebBNPool2(nn.Module):
    """Building Block with a pooling/unpooling and a Chebyshev Convolution.
    """

    def __init__(self, in_channels, middle_channels, out_channels, lap, pooling, kernel_size, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree.
        """
        super().__init__()
        self.pooling = pooling
        self.spherical_cheb = BayesianSphericalChebBN2(in_channels, middle_channels, out_channels, lap, kernel_size, weight_regularizer, dropout_regularizer)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.Tensor`: output [batch x vertices x channels/features]
        """
        x = self.pooling(x)
        x = self.spherical_cheb(x)
        return x