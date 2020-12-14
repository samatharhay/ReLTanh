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
    def __init__(self, alpha=0.0, beta=-1.5, learnable=False):
        super(ReLTanh, self).__init__()

        assert alpha > beta

        self.alpha = alpha
        self.beta = beta
        self.alpha_t = torch.FloatTensor([self.alpha])
        self.beta_t = torch.FloatTensor([self.beta])

        self.learnable = learnable

        if self.learnable:
            self.alpha_t = Parameter(self.alpha_t, requires_grad=True)
            self.beta_t = Parameter(self.beta_t, requires_grad=True)

    def __repr__(self):
        if self.learnable:
            return f"ReLTanh(alpha={self.alpha_t.data.item()}, beta={self.beta_t.data.item()}, learnable={self.learnable})"
        return f"ReLTanh(alpha={self.alpha}, beta={self.beta})"

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        # move variables to correct device
        device = x.device
        self.alpha_t = self.alpha_t.to(device)
        self.beta_t = self.beta_t.to(device)

        # Set up boundary values
        alpha_tanh = torch.tanh(self.alpha_t)
        beta_tanh = torch.tanh(self.beta_t)
        one = torch.ones([1]).to(device)
        alpha_tanh_d1 = one.sub(torch.square(alpha_tanh))
        beta_tanh_d1 = one.sub(torch.square(beta_tanh))

        # compute masks to relax indifferentiability
        alpha_mask = x.ge(self.alpha_t)
        beta_mask = x.le(self.beta_t)
        act_mask = ~(alpha_mask | beta_mask)

        # activations
        x_alpha = x.sub(self.alpha_t).mul(alpha_tanh_d1).add(self.alpha_t)
        x_beta = x.sub(self.beta_t).mul(beta_tanh_d1).add(self.beta_t)
        x_act = torch.tanh(x)

        # combine activations
        x = x_alpha.mul(alpha_mask) + x_beta.mul(beta_mask) + x_act.mul(act_mask)

        return x
