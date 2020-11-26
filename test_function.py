# Import basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict

# Import PyTorch
import torch # import main library
from torch.autograd import Variable
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations
import torch.nn.functional as F # import torch functions
from torchvision import datasets, transforms # import transformations to use for demo


class ReLTanh(nn.Module):
    '''
    Implementation of ReLTanh activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - constant parameter
        - beta - constant parameter
    References:
        - See related paper:
        https://www.sciencedirect.com/science/article/pii/S0925231219309464
    Examples:
        >>> a1 = ReLTanh(0, -1.5)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, alpha=0.0, beta=-1.5):
        super(ReLTanh, self).__init__()

        assert alpha > beta

        self.alpha = torch.FloatTensor([alpha])
        self.beta = torch.FloatTensor([beta])

        # Set up constant boundary values
        self.alpha_tanh = torch.tanh(self.alpha)
        self.beta_tanh = torch.tanh(self.beta)
        self.alpha_tanh_d1 = torch.ones([1]).sub(torch.square(self.alpha_tanh))
        self.beta_tanh_d1 = torch.ones([1]).sub(torch.square(self.beta_tanh))

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''

        # compute masks to relax indifferentiability
        alpha_mask = x.ge(self.alpha)
        beta_mask = x.le(self.beta)
        act_mask = ~(alpha_mask | beta_mask)

        # activations
        x_alpha = x.sub(self.alpha).mul(self.alpha_tanh_d1).add(self.alpha)
        x_beta = x.sub(self.beta).mul(self.beta_tanh_d1).add(self.beta)
        x_act = torch.tanh(x)

        # combine activations
        x = x_alpha.mul(alpha_mask) + x_beta.mul(beta_mask) + x_act.mul(act_mask)

        return x
