import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##import accuracy

accuracy_csv = {
    'RelTanh' : "accuracy/ReLTanh_5h_30e.csv",
    'Tanh' : "accuracy/Tanh_5h_30e.csv"
}
for name in accuracy_csv:
    x = np.array(pd.read_csv(accuracy_csv[name],header = None,index_col=None).values).flatten()
    plt.plot(np.arange(len(x)),x,label = name)
    #print(x)

plt.legend()
plt.show()
##import loss

accuracy_csv = {
    'RelTanh' : "loss/ReLTanh_5h_30e.csv",
    'Tanh' : "loss/Tanh_5h_30e.csv"
}
for name in accuracy_csv:
    x = np.array(pd.read_csv(accuracy_csv[name],header = None,index_col=None).values).flatten()
    plt.plot(np.arange(len(x)),x,label = name)
    #print(x)

plt.legend()
plt.show()

