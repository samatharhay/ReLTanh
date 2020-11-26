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
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = None, beta = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(ReLTanh,self).__init__()
        self.in_features = in_features

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.tensor(alpha)) # create a tensor out of alpha
            
        if beta == None:
            self.beta = Parameter(torch.tensor(-1.5))
        else:
            self.beta = Parameter(torch.tensor(beta))
                
            
        self.alpha.requiresGrad = True # set requiresGrad to true!
        self.beta.requiresGrad = True 

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        for i in range(len(x)):
            for j in range(len(x[0])):
                if x[i][j] > self.beta and x[i][j] < self.alpha:
                    x[i][j] = torch.tanh(x[i][j])
                    
                elif x[i][j]>= self.alpha:
                    x[i][j] = (1-torch.tanh(self.alpha)**2)*(x[i][j]-self.alpha)+torch.tanh(self.alpha)
                elif x[i][j]<= self.beta:
                    x[i][j] = (1-torch.tanh(self.beta)**2)*(x[i][j]-self.beta)+torch.tanh(self.beta)
                else:
                    print('Error: ReLTanh',x[i][j])
        return x