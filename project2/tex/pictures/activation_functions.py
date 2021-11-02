"""
This file produces plots of the different activation functions
which are displayed in the report.
"""
import numpy as np
import matplotlib.pyplot as plt

#The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')


def sigmoid(x):
    """
    Returns the Sigmoid activation function
    """
    return 1/(1+np.exp(-x))


def RELU(x):
    """
    Returns the RELU activation function
    """
    return np.maximum(0,x)


def Leaky_RELU(x):
    """
    Returns the Leaky RELU activation function
    """
    return np.where(x > 0, x, x * 0.1)

fig, axs = plt.subplots(1,3)
x = np.linspace(-5, 5, 1000)
axs[0].plot(x, sigmoid(x))
axs[0].set_title('Sigmoid', fontsize=20)
axs[1].plot(x, RELU(x))
axs[1].set_title('RELU', fontsize=20)
axs[2].plot(x, Leaky_RELU(x))
axs[2].set_title('Leaky RELU', fontsize=20)
for i in range(len(axs)):
    axs[i].tick_params(axis='both', which='major', labelsize=15)
    axs[i].grid(False)
fig.set_size_inches(8,2)
fig.tight_layout()
fig.savefig('activation_functions.pdf')
