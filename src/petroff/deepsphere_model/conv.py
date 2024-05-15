"""Chebyshev convolution layer. For the moment taking as-is from Michaël Defferrard's implementation. For v0.15 we will rewrite parts of this layer.
"""

# From https://github.com/deepsphere/deepsphere-pytorch/blob/master/deepsphere/layers/chebyshev.py

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
# Changes: Added notes about the need to replace the initialization method.

# pylint: disable=W0221

import math

import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.
    
    This function applies a convolution operation using Chebyshev polynomials 
    up to degree K-1, leveraging the graph Laplacian's spectral properties. 
    The convolution is designed to encode both the graph structure and 
    node features into the learning process efficiently.

    Recall that the definition of the Chebyshev polynomials (of the first kind) is

    T_{0}(x) = 1
    T_{1}(x) = x
    T_{n}(x) = 2x T_{n-1}(x) - T_{n-2}(x)
    
    e.g.
    T_{2}(x) = 2x^2 - 1
    T_{3}(x) = 4x^3 - 3x
    T_{4}(x) = 8x^4 - 8x^2 + 1

    Note that $x$ here is the laplacian.

    Note also that the parameter K (we use degree K-1 polynomials) is implicitely stored
    in the shape of the weights.

    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.

    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    B, V,   Fin  = inputs.shape                                # B x V x Fin
    K, Fin, Fout = weight.shape                                # K x Fin x Fout
    # B    = batch size
    # V    = nb vertices
    # Fin  = nb input features
    # Fout = nb output features
    # K    = order of Chebyshev polynomials

    # transform to Chebyshev basis
    # T_{0}(x) = 1, constant, so we have no Laplacian
    x0 = inputs.permute(1, 2, 0).contiguous()                   #     V x Fin x B
    x0 = x0.view([V, Fin * B])                                  #     V x Fin * B
    inputs = x0.unsqueeze(0)                                    # 1 x V x Fin * B

    if K > 0:
        # T_{1}(x) = x; the first application of the laplacian to the data
        x1     = torch.sparse.mm(laplacian, x0)                 #     V x Fin * B
        # We have two terms in the recursive definition; we concatenate T_{0} and T_{1}
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)        # 2 x V x Fin * B
        # Now we proceed to the degree K desired
        for _ in range(1, K - 1):
            # T_{n} is 2 * T_{n-1}(x)                   - T_{n-2}(x)
            x2     = 2 * torch.sparse.mm(laplacian, x1) - x0
            # Concatenate the new result
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0)    #     K x Fin * B
            # We systematically increase the order of the polynomials applied:
            x0, x1 = x1, x2

    # We now have successive applications of the Laplacian to the data
    #    Rearrange to have the expected order
    inputs = inputs.view([K, V, Fin, B])                        # K x V x Fin x B
    inputs = inputs.permute(3, 1, 2, 0).contiguous()            # B x V x Fin x K
    inputs = inputs.view([B * V, Fin * K])                      # B * V x Fin * K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout)

    # And we can then apply the weights stored in the model
    inputs = inputs.matmul(weight)                              # B * V x Fout
    inputs = inputs.view([B, V, Fout])                          # B x V x Fout

    return inputs


class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, conv=cheb_conv):
        """Initialize the Chebyshev layer.

        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
            conv (callable): Function which will perform the actual convolution.
        """
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self._conv        = conv

        shape = (kernel_size, in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # TODO: Replace with Petroff's initialization method
        # self.kaiming_initialization()
        self.xavierlike_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        logger.warning("Using Kaiming Initialization! Need to use Xavier-like initialization instead.")
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def xavierlike_initialization(self):
        """Initialize weights and bias using Xavier-like initialization."""
        # Xavier-like weight initializer for Chebyshev coefficients
        std = 1 / math.sqrt(self.in_channels * (self.kernel_size + 0.5) / 2)
        nn.init.normal_(self.weight, 0, std)
        nn.init.uniform_(self.bias)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.

        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.

        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs


class SphericalChebConv(nn.Module):
    """Building Block with a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size):
        """Initialization.

        Args:
            in_channels (int): initial number of channels
            out_channels (int): output number of channels
            lap (:obj:`torch.sparse.FloatTensor`): laplacian
            kernel_size (int): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.register_buffer("laplacian", lap)
        self.chebconv = ChebConv(in_channels, out_channels, kernel_size)

    # def state_dict(self, *args, **kwargs):
    #     """! WARNING !

    #     This function overrides the state dict in order to be able to save the model.
    #     This can be removed as soon as saving sparse matrices has been added to Pytorch.
    #     """
    #     state_dict = super().state_dict(*args, **kwargs)
    #     del_keys = []
    #     for key in state_dict:
    #         if key.endswith("laplacian"):
    #             del_keys.append(key)
    #     for key in del_keys:
    #         del state_dict[key]
    #     return state_dict

    def forward(self, x):
        """Forward pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.chebconv(self.laplacian, x)
        return x