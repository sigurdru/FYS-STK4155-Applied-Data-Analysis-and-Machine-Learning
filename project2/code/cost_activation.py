"""
This file contains cost and activation functions used in the code
"""
import autograd.numpy as np

class Costs:
    def MSE(self, t_):
        mse = (t_ - self.t)**2 / len(self.t)
        return mse

    def cross_entropy(self, t_):
        a = -(self.t * np.log(t_) + (1 - self.t) * np.log(1 - t_))
        return a


class Activations:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.where(x > 0, x, 0)

    def leaky_relu(self, x, leak=0.01):
        return np.where(x > 0, x, leak * x)

    def softmax(self, x):
        exp = np.exp(x)
        s = np.sum(exp, axis=1, keepdims=True)
        return exp/s

    def linear(self, x):
        # No activation
        return x
