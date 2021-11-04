import autograd.numpy as np
from tqdm import tqdm
from autograd import elementwise_grad
from cost_activation import Costs, Activations
import utils 


class Optimizer:
    """
    Learning optimizer for neural network.
    Set gamma to give more or less momentum
    """
    def __init__(self, gamma=0, layer_sizes=None):
        self.g = min(1, max(gamma, 0))  # ensure between 0 and 1
        self.prev_v = [np.zeros(i) for i in layer_sizes]

    def __call__(self, eta_grad, layer=0):
        v = self.g * self.prev_v[layer] + eta_grad
        self.prev_v[layer] = v
        return v


class FFNN(Costs, Activations):  
    """
    Feed Forward Neural Network

    from cost_activations.py:
    Costs, Activations
     - contains different cost and activation functions 
       used for backpropagation 
    """
    def __init__(self,
                 design,
                 target,
                 hidden_nodes=[10,],
                 batch_size=None,
                 learning_rate=0.1,
                 lmb=0,
                 gamma=0,
                 activation="sigmoid",
                 cost="MSE",
                 output_activation="none",
                 test_data=None,
                 ):

        self.X = design             # Training data
        self.N = self.X.shape[0]    # Number of input values
        self.static_target = target # Trainging outputs

        # Set mini batch size
        if batch_size in [0, None]:
            self.batch_size = self.N
        else:
            self.batch_size = batch_size
        self.test_data = test_data

        self.eta = learning_rate
        self.lmb = lmb
        bias0 = 0.01  # initial bias value

        # Number of nodes in all layers 
        self.nodes = np.array([self.X.shape[1], *hidden_nodes, self.static_target.shape[1]])

        # All layers of neural network (1 input, n hidden, 1 output)
        # Number of rows in each layer corresponds to the number of data points   
        self.Layers = [np.zeros((self.N, n)) for n in self.nodes]
        self.Layers[0] = self.X.copy()

        self.z = self.Layers.copy()         # Activation layer input
        self.delta_l = self.Layers.copy()   # Activation layer input gradient

        # Initial zero for weights ensures that weights[n] corresponds to Layers[n]
        self.weights = [0] + [np.random.normal(scale=1 / n, size=(n, m)) for n, m in zip(self.nodes[:-1], self.nodes[1:])]
        self.bias = [np.ones((1, n)) * bias0 for n in self.nodes]

        # Calculate gradients with momentum (gamma=0 by default)
        self.optim_w = Optimizer(gamma, self.nodes)
        self.optim_b = Optimizer(gamma, self.nodes)

        # Activation functions available
        activation_funcs = {'sigmoid': self.sigmoid,
                            'tanh': self.tanh,
                            'relu': self.relu,
                            'leaky_relu': self.leaky_relu,
                            'softmax': self.softmax,
                            'none': self.none}

        # Cost functions avaible
        cost_funcs = {'MSE': self.MSE,
                      'cross_entropy': self.cross_entropy}

        # Callable activation and cost function and their derivatives
        self.activation = activation_funcs[activation]
        self.activation_out = activation_funcs[output_activation]
        self.cost = cost_funcs[cost]

        self.activation_der = elementwise_grad(self.activation)
        self.out_der = elementwise_grad(self.activation_out)
        self.cost_der = elementwise_grad(self.cost)


    def backpropagation(self):
        """
        Updates weights and biases with backwards propagation.
         - Starts by calculating the gradients of each layer's activation function
         - Updates the weights and biases accordingly
        """
        # Calculate gradient of output layer
        if self.activation_out.__name__ == 'softmax' and self.cost.__name__ == 'cross_entropy':
            # Analytical derivative for softmax and cross entropy 
            self.delta_l[-1] = self.Layers[-1] - self.t

        elif self.activation_out.__name__ == 'none':
            self.delta_l[-1] = self.cost_der(self.Layers[-1])

        else:
            self.delta_l[-1] = self.cost_der(self.Layers[-1]) * self.out_der(self.z[-1])

        # Calculate gradients of the hidden layers
        for i in reversed(range(1, len(self.nodes) - 1)):
            self.delta_l[i] = self.delta_l[i + 1] @ self.weights[i + 1].T \
                                * self.activation_der(self.z[i])

        # Update weights and biases for each previous layer
        for n in reversed(range(1, len(self.nodes))):
            # find weight gradient with l2-norm
            weight_gradient = self.Layers[n - 1].T @ self.delta_l[n] + self.lmb * self.weights[n] #/ len(self.z[n])
            self.weights[n] -= self.optim_w(self.eta * weight_gradient, n)
            self.bias[n] -= self.optim_b(self.eta * np.sum(self.delta_l[n], axis=0), n)


    def feed_forward(self):
        # Update the hidden layers
        for n in range(1, len(self.nodes)):
            self.z[n] = self.Layers[n - 1] @ self.weights[n] + self.bias[n]
            self.Layers[n] = self.activation(self.z[n])

        self.Layers[-1] = self.activation_out(self.z[n]) # Update final layer 


    def train(self, epochs, train_history=False):
        """
        Training the neural network by looping over epochs:
         1) Initializing shuffled minibatches
         2) Update each layers with their weights and biases
         3) Updates weights and biases with backpropagation
        """

        history = np.zeros(epochs)
        errors = np.zeros(epochs)

        indicies = np.arange(self.N)
        pbar = tqdm(range(epochs), desc=f"eta: {self.eta}, lambda: {self.lmb}. Training")

        for epoch in pbar:

            if train_history:
                if self.test_data is None:
                    t = self.static_target
                    x = self.X
                else:
                    t = self.test_data[1]
                    x = self.test_data[0]
                self.t = t
                output = self.predict(x)
                # print(np.c_[output, np.sum(output, axis=1)])
                pred = np.argmax(output, axis=1).reshape(-1,1)
                loss = output - self.t

                if self.nodes[-1] > 1:
                    r = np.argmax(self.t, axis=1).reshape(-1,1)
                    history[epoch] = np.sum(pred == r) / len(r)
                    errors[epoch] = np.mean(loss, axis=0)[0]
                else:
                    history[epoch] = np.sum(self.cost(pred), axis=1)
                
            np.random.shuffle(indicies)  # Shuffle indices

            self.X_s = self.X[indicies]  # Shuffled input
            self.shuffle_t = self.static_target[indicies]  # Shuffled target

            for i in range(0, self.N, self.batch_size):
                # Loop over minibatches
                self.Layers[0] = self.X_s[i: i + self.batch_size]
                self.t = self.shuffle_t[i: i + self.batch_size]

                self.feed_forward()
                self.backpropagation()

        return history, errors


    def predict(self, x):
        """
        input: x (ndarray)
        Calculate output layer with updated weights and biases
        """
        self.Layers[0] = x
        self.feed_forward()
        # if self.nodes[-1] > 1:
            # return np.argmax(self.Layers[-1], axis=1).reshape(-1,1)
        return self.Layers[-1]

    def save(self, fname=None):
        """
        Save weights and biases of trained network  
        """
        if fname is None:
            fname = f"a{self.activation.__name__}_o{self.activation_out.__name__}_c{self.cost.__name__}"
            for i in self.nodes:
                fname += "_" + str(i)
        data = {"weights": self.weights, "biases": self.bias}
        fname += + ".npy" if ".npy" not in fname else ""
        np.save("../output/saved_nets/" + fname, data)

    def load(self, fname):
        """
        Load weights and biases from trained network 
        """
        fname += ".npy" if ".npy" not in fname else ""
        data = np.load("../output/saved_nets/" + fname, allow_pickle=True).item()
        self.weights = data["weights"]
        self.bias = data["biases"]


if __name__ == "__main__":
    from utils import *
    from sklearn.model_selection import train_test_split
    np.random.seed(2021)

    class Args:
        num_points = 20
        epsilon = 0.2
        polynomial = 1
        dataset = "Franke"
        bs = 32
    args = Args()
    epochs = 1000

    x, y, z = load_data(args)
    X = create_X(x, y, args.polynomial, intercept=False)
    X_, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    MM = FFNN(X_,
              z_train,
              hidden_nodes=[16],
              batch_size=args.bs,
              learning_rate=0.1,
              lmb=1e-3,
              gamma=0.2,
              activation="sigmoid",
              cost="MSE")

    MM.train(epochs)
    print(np.c_[MM.predict(X_), z_train])
    nn_pred = MM.predict(X_test)

    print("Neural Network")
    print("    MSE:", MSE(z_test, nn_pred))
    print("    R2-score:", R2(z_test, nn_pred))
