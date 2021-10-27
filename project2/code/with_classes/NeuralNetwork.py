import numpy as np
import autograd.numpy as anp 
from tqdm import tqdm
from autograd import elementwise_grad, grad 

class FFNN:  # FeedForwardNeuralNetwork

    def __init__(self,
                 design,
                 target,
                 hidden_nodes=[10,],
                 batch_size=1,
                 learning_rate=0.1,
                 lmb=0,
                 activation="sigmoid",
                 cost="MSE",
                 ):
        self.X = design
        # print(self.X.shape)
        self.t = target
        self.static_target = target.copy()
        self.N = self.X.shape[0]
        inputs = self.X.shape[1]
        outputs = self.t.shape[1]

        self.batch_size = self.N
        self.eta = learning_rate
        self.lmb = lmb
        bias0 = 0.01

        self.num_nodes = [inputs,] + list(hidden_nodes) + [outputs,]

        self.layer_activations = [np.zeros((self.N, n)) for n in self.num_nodes]
        self.layer_activations[0]  = self.X.copy()
        self.layer_inputs = self.layer_activations.copy()
        self.lgradient = self.layer_activations.copy()

        self.weights = [np.random.randn(n, m) for n, m in zip(self.num_nodes[1:], self.num_nodes[:-1])]
        self.bias = [np.ones((1, n)) * bias0 for n in self.num_nodes]
        # for w in self.layer_activations:
            # print(w.shape)
        # print(len(self.weights[1].T))

        self.activation = self.sigmoid
        self.cost = self.MSE
        self.activation_der = elementwise_grad(self.activation)#(0.2)
        self.cost_der = elementwise_grad(self.cost)
        # print(self.activation)
        # grad_sigmoid = grad(self.sigmoid)
        # print(self.activation_der(0.2))
        # print(grad_sigmoid(1.0))
        # exit()

    def update(self):
        self.backprop()

        for n in range(1, len(self.num_nodes)):
            a = self.lgradient[n].T @ self.layer_activations[n - 1]
            # print(len(self.lgradient[n].T[0]))
            # print(self.eta * (self.lgradient[n].T @ self.layer_activations[n - 1]) / self.batch_size)
            # print(self.batch_size, self.eta)
            # print(len(a))
            # print(len(self.weights[n][0]))
            # exit()
            self.weights[n-1] -= self.eta * (self.lgradient[n].T @ self.layer_activations[n - 1]) / self.batch_size
            self.bias[n] -= self.eta * np.mean(self.lgradient[n].T, axis=1)
    
    def feed_forward(self):
        for n in range(1, len(self.num_nodes)):
            # print(self.layer_activations[n - 1].shape)
            # print(self.weights[n-1].T.shape)
            # print(self.bias[n].shape)
            z_h = self.layer_activations[n - 1] @ self.weights[n-1].T + self.bias[n]
            self.layer_inputs[n] = z_h
            self.layer_activations[n] = self.activation(z_h)

    def backprop(self):
        self.lgradient[-1] = self.cost_der(self.layer_activations[-1]) * self.activation_der(self.layer_inputs[-1])
        for n in reversed(range(1, len(self.num_nodes) - 1)):
            # print(n, (self.weights[n]))
            self.lgradient[n] = self.lgradient[n + 1] @ self.weights[n] * self.activation_der(self.layer_inputs[n])
            # exit()

    def predict(self, x):
        self.layer_activations[0] = x
        self.feed_forward()
        return self.layer_activations[-1]
    
    def train(self, epochs):
        pbar = tqdm(range(epochs), desc="Training epochs")
        for _ in pbar:
            self.SGD()
            self.feed_forward()
            self.update()
            pbar.update(1)

    def SGD(self):
        inds = np.random.choice(np.arange(self.N), size=self.batch_size, replace=False)
        self.layer_activations[0] = self.X[inds]
        self.t = self.static_target[inds]

    def sigmoid(self, x):
        y = 1 / (1 + anp.exp(-x))
        return y#1 / (1 + np.exp(-x))

    def MSE(self, t_):
        return (t_ - self.t) ** 2