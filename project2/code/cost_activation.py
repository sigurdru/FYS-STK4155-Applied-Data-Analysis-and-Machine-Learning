import autograd.numpy as np


class Costs:
    def MSE(self, t_):
        mse = (t_ - self.t)**2 / len(self.t)
        return mse

    def accuracy_score(self, t_):
        raise NotImplementedError()


class Activations:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.where(x > 0, x, 0)

    def leaky_relu(self, x):
        return np.where(x > 0, x, 0.01*x)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
