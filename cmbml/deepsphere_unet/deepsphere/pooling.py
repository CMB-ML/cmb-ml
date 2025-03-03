import torch.nn as nn
import torch.nn.functional as F

class HealpixMaxPool(nn.MaxPool1d):
    def __init__(self, return_indices=False):
        """Initialization
        """
        super().__init__(kernel_size=4, return_indices=return_indices)

    def forward(self, x):
        """Forward call the 1d Maxpooling of pytorch

        Args:
            x (:obj:`torch.tensor`):[batch x pixels x features]

        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [batch x pooled pixels x features] and indices of pooled pixels
        """
        x = x.permute(0, 2, 1)
        if self.return_indices:
            x, indices = F.max_pool1d(x, self.kernel_size)
        else:
            x = F.max_pool1d(x, self.kernel_size)
        x = x.permute(0, 2, 1)

        if self.return_indices:
            output = x, indices
        else:
            output = x
        return output

class HealpixNNUsample(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest'):
        """Initialization
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
    
    def forward(self, x):
        """Forward call the 1d Maxpooling of pytorch

        Args:
            x (:obj:`torch.tensor`):[batch x pixels x features]

        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [batch x pooled pixels x features] and indices of pooled pixels
        """
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        return x

class Pool:
    def __init__(self):
        self.__pool = HealpixMaxPool()
        self.__upsample = HealpixNNUsample(scale_factor=4)

    @property
    def pool(self):
        return self.__pool
    
    @property
    def upsample(self):
        return self.__upsample