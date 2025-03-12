## Code is taken and modified from the following repo: https://github.com/aurelio-amerio/ConcreteDropout.git
import torch
import torch.nn as nn


class SpatialConcreteDropout(nn.Module):
    """ 
    Spatial Concrete Dropout layer as proposed in https://arxiv.org/abs/1705.07832
    This module allows the model to automatically learn the dropout probability, p, during training.
    The spatial variation allows the regularization to be adapted to convolutional layers.
    """
    def __init__(self, channels_first: bool=False, weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.001, init_max=0.001, is_mc_dropout=False):
        """Initializes the Spatial Concrete Dropout layer.

        Args:
            channels_first (bool, optional): The dropout layers and regularizations are applied to the channels, this determines the shape needed. Defaults to False.
            weight_regularizer (float, optional): A positive number which satisfies
                                                    $weight_regularizer = l**2 / (\tau * N)$
                                                with prior lengthscale l, model precision $\tau$ (inverse observation noise),
                                                and N the number of instances in the dataset. Defaults to 1e-6.
            dropout_regularizer (float, optional):  A positive number which satisfies
                                                        $dropout_regularizer = 2 / (\tau * N)$
                                                    with model precision $\tau$ (inverse observation noise) and N the number of
                                                    instances in the dataset. Defaults to 1e-5.
            Note: The regularization terms are used to compute the KL divergence loss term.
                    The relation between the terms is
                    $weight_regularizer / dropout_regularizer = l**2 / 2$
                    with prior lengthscale l. Note also that the factor of two should be
                    ignored for cross-entropy loss, and used only for the euclidean loss.
            init_min (float, optional): Lower bound for uniform distribution initialization. Defaults to 0.001.
            init_max (float, optional): Upper bound for uniform distribution initialization. Defaults to 0.001.
        """
        super(SpatialConcreteDropout, self).__init__()
        self.channels_first = channels_first
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min = torch.log(torch.tensor(init_min / (1 - init_min)))
        init_max = torch.log(torch.tensor(init_max / (1 - init_max)))
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)
        self.is_mc_dropout = is_mc_dropout
        # Buffers to train on GPU, otherwise will be mad
        self.register_buffer('eps', torch.tensor(torch.finfo(torch.float32).eps))
        self.register_buffer('temperature', torch.tensor(2. / 3.))

    def _get_noise_shape(self, inputs):
        input_shape = inputs.shape
        if self.channels_first:
            return (input_shape[0], input_shape[1], 1)
        else:
            return (input_shape[0], 1, input_shape[2])

    def _concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        p = self.p
        # machine precision epsilon for numerical stability inside the log
        eps = self.eps

        # this is the shape of the dropout noise, same as tf.nn.dropout
        noise_shape = self._get_noise_shape(x)

        unif_noise = torch.rand(*noise_shape, device=p.device)  # uniform noise
        # bracket inside equation 5, where u=uniform_noise
        drop_prob = (
            torch.log(p + eps)
            - torch.log1p(eps - p)
            + torch.log(unif_noise + eps)
            - torch.log1p(eps - unif_noise)
        )
        drop_prob = torch.sigmoid(drop_prob / self.temperature)  # z of eq 5
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x = torch.mul(x, random_tensor)  # we multiply the input by the concrete dropout mask
        x /= retain_prob  # we normalise by the probability to retain

        return x

    def get_regularization(self, x, layer):
        p = self.p
        # We will now compute the KL terms following eq.3 of 1705.07832
        weight = layer.weight
        ss = torch.sum(torch.pow(weight, 2))

        # The kernel regularizer corresponds to the first term
        # Note: we  divide by (1 - p) because  we  scaled  layer  output  by (1 - p)
        kernel_regularizer = self.weight_regularizer * ss / (1. - p)
        # the dropout regularizer corresponds to the second term
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer = dropout_regularizer + ((1. - p) * torch.log(1 - p))
        if self.channels_first:
            input_dim = x.shape[1]
        else:
            input_dim = x.shape[-1]

        dropout_regularizer = dropout_regularizer * (self.dropout_regularizer * input_dim)
        # this is the KL term to be added as a loss
        # regularizer
        return torch.sum(kernel_regularizer + dropout_regularizer)

    def forward(self, x, layer):
        self.p = torch.sigmoid(self.p_logit)
        self.regularization = self.get_regularization(x, layer)
        if self.is_mc_dropout:
            return layer(self._concrete_dropout(x))
        else:
            if self.training:
                return layer(self._concrete_dropout(x))
            else:
                return layer(x)