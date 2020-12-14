import torch
import numpy as np
import matplotlib.pyplot as plt
import math


def ReLTanh(x, alpha, beta):
    for i in range(x.size):
        if x[i]> beta and x[i] < alpha:
            x[i] = np.tanh(x[i])
        elif x[i]>= alpha:
            x[i] = (1-np.tanh(alpha)**2)*(x[i]-alpha)+np.tanh(alpha)
        elif x[i]<= beta:
            x[i] = (1-np.tanh(beta)**2)*(x[i]-beta)+np.tanh(beta)
        else:
            print('Error: ReLTanh',x[i])
    return x

def ddtanh(x):
    return 8*((math.e**(-2*x)) - (math.e**(2*x)))/(((math.e**x)+(math.e**-x))**4)


def derivative_ReLTanh(x, alpha, beta):
    for i in range(x.size):
        if x[i] > beta and x[i] < alpha:
            x[i] = 1-np.tanh(x[i])**2
        elif x[i]>= alpha:
            #x[i] = (1-np.tanh(alpha)**2)
            x[i] = ddtanh(alpha)
        elif x[i]<= beta:
            #x[i] = (1-np.tanh(beta)**2)
            x[i] = ddtanh(beta)
        else:
            print('Error: ReLTanh',x[i])
    return x


x = np.arange(-6.0,6.1,.1)
y = ReLTanh(x,0,-1.5)
t = np.arange(-6.0,6.1,.1)
p = ReLTanh(t,.5,np.NINF)
q = np.arange(-6.0,6.1,.1)
plt.plot(q,y, label = '(0,-1.5)')
plt.plot(q,p, label = '(.5, -inf)')
plt.legend()
plt.show()

x = np.arange(-6.0,6.1,.1)
y = derivative_ReLTanh(x,0,-1.5)
t = np.arange(-6.0,6.1,.1)
p = derivative_ReLTanh(t,.5,np.NINF)
q = np.arange(-6.0,6.1,.1)
plt.plot(q,y, label = '(0,-1.5)')
plt.plot(q,p, label = '(.5, -inf)')
plt.legend()
plt.show()