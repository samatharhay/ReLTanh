import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##import accuracy

accuracy_csv = {
    'ReLTanh(λ+=0, λ−= -1.5)' : "logs/cnn_cifar10_reltanh_8l_300e_a_0.0_b_-1.5/accuracy/train.csv",
    'ReLTanh(λ+= 0.5, λ−= -inf)' : "logs/cnn_cifar10_reltanh_8l_300e_a_0.5_b_-1e+29/accuracy/train.csv",
    'ReLU' : "logs/cnn_cifar10_relu_8l_300e/accuracy/train.csv",
    'Tanh' : "logs/cnn_cifar10_tanh_8l_300e/accuracy/train.csv"
    
}
for name in accuracy_csv:
    x = np.array(pd.read_csv(accuracy_csv[name],header = None,index_col=None).values).flatten()
    plt.plot(np.arange(len(x)),x,label = name)
    #print(x)
plt.ylabel("Training Accuracy (%)")
plt.xlabel("Epoch")
plt.legend()
plt.show()
##import loss

loss_csv = {
    'RelTanh_0_-1.5' : "logs/cnn_cifar10_reltanh_8l_300e_a_0.0_b_-1.5/loss/train.csv",
    'RelTanh_.5_-INF' : "logs/cnn_cifar10_reltanh_8l_300e_a_0.5_b_-1e+29/loss/train.csv",
    'ReLU' : "logs/cnn_cifar10_relu_8l_300e/loss/train.csv",
    'Tanh' : "logs/cnn_cifar10_tanh_8l_300e/loss/train.csv"
}
for name in loss_csv:
    x = np.array(pd.read_csv(loss_csv[name],header = None,index_col=None).values).flatten()
    plt.plot(np.arange(len(x)),x,label = name)
    #print(x)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

