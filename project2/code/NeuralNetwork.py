"""
Contains code for neural network, as well as optimizer (momentum sgd)
"""
from collections import defaultdict
import autograd.numpy as np
from tqdm import tqdm
from autograd import elementwise_grad
from cost_activation import Costs, Activations
import utils
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)



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
                 args,
                 hidden_nodes=[10,],
                 batch_size=None,
                 learning_rate=0.1,
                 dynamic_eta=False,
                 lmb=0,
                 gamma=0,
                 wi=True,
                 activation="sigmoid",
                 cost="MSE",
                 output_activation="none",
                 ):

        self.X = design             # Training data
        self.N = self.X.shape[0]    # Number of input values
        self.static_target = target # Trainging outputs
        self.args = args 

        if self.args.dataset == "Franke" and self.args.history:
            # Used for rescaling Franke data when calculating MSE in training history
            self.z_Franke = utils.load_data(args)[-1]


        # Set mini batch size
        if batch_size in [0, None]:
            self.batch_size = self.N
        else:
            self.batch_size = batch_size

        self.eta0 = learning_rate
        self.de = dynamic_eta
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
        if wi and activation not in ("none", "linear"):
            if activation == "sigmoid":  # Xavier initialization
                self.weights = [0] + [np.random.uniform(-1/np.sqrt(n), 1/np.sqrt(n), size=(n, m)) for n, m in zip(self.nodes[:-1], self.nodes[1:])]
            elif activation == "tanh":  # Normalized Xaviet initialization
                self.weights = [0] + [np.random.uniform(-np.sqrt(6/(n + m)), np.sqrt(6/(n + m)), size=(n, m)) for n, m in zip(self.nodes[:-1], self.nodes[1:])]
            elif activation == "relu":  # He initialization
                self.weights = [0] + [np.random.normal(scale=np.sqrt(2/n), size=(n, m)) for n, m in zip(self.nodes[:-1], self.nodes[1:])]
            elif activation == "leaky_relu":  # He initialization
                self.weights = [0] + [np.random.normal(scale=np.sqrt(2/(1 + 0.01**2)/n), size=(n, m)) for n, m in zip(self.nodes[:-1], self.nodes[1:])]
        else: # no fancy initialization
            self.weights = [0] + [np.random.normal(size=(n, m)) for n, m in zip(self.nodes[:-1], self.nodes[1:])]
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
                            'none': self.linear,
                            'linear': self.linear,
                            }

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
         - Returns wether or not the weights converged
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
            # find weight and bias gradient with l2-norm
            weight_gradient = self.Layers[n - 1].T @ self.delta_l[n] + self.lmb * self.weights[n]
            bias_grad = np.sum(self.delta_l[n], axis=0) + self.lmb * self.bias[n]
            # calculate the weight and bias update
            weight_change = self.optim_w(self.eta * weight_gradient, n)
            bias_change = self.optim_b(self.eta * bias_grad, n)
            # update weight and bias
            self.weights[n] -= weight_change
            self.bias[n] -= bias_change

        # If weight and bias to output layer doesnt change, end training
        if max(np.max(abs(weight_change)), np.max(abs(bias_change))) < 1e-8:
            self.converged = True

    def feed_forward(self):
        # Update the hidden layers
        for n in range(1, len(self.nodes)):
            self.z[n] = self.Layers[n - 1] @ self.weights[n] + self.bias[n]
            self.Layers[n] = self.activation(self.z[n])

        self.Layers[-1] = self.activation_out(self.z[n]) # Update final layer

    def train(self, epochs, train_history=False, test=None):
        """
        Training the neural network by looping over epochs:
         1) Initializing shuffled minibatches
         2) Update each layers with their weights and biases
         3) Updates weights and biases with backpropagation

        Arguments:
            epochs: int
                total number of epochs to train
            train_history: bool
                if true, calculate mse/accuracy during training, on train data
            test: 2tuple with X_test and z_test
                is not None, also calculate mse/accuracy during training on test data
        """
        self.history = defaultdict(lambda: np.zeros(epochs) * np.nan)
        self.converged = False
        indicies = np.arange(self.N)
        pbar = tqdm(range(epochs), desc=f"eta: {self.eta0}, lambda: {self.lmb}. Training")

        for epoch in pbar:

            # Learning schedule
            self.eta = self.eta0 * (1 - epoch / epochs) if self.de else self.eta0

            # save training performance
            if train_history:
                self.train_history(epoch, test)

            np.random.shuffle(indicies)  # Shuffle indices
            self.X_s = self.X[indicies]  # Shuffled input
            self.shuffle_t = self.static_target[indicies]  # Shuffled target

            for i in range(0, self.N, self.batch_size):
                # Loop over minibatches
                self.Layers[0] = self.X_s[i: i + self.batch_size]
                self.t = self.shuffle_t[i: i + self.batch_size]

                try:
                    self.feed_forward()
                    self.backpropagation()
                except RuntimeWarning:
                    self.converged = None
                    break

            pbar.set_description(f"eta: {self.eta:.3f}, lambda: {self.lmb:.3f}. Training")
            if self.converged:
                print(f"Network converged after {epoch} epochs")
                break
            elif self.converged is None:
                print("RuntimeWarning. Overflow or underflow encoutered")
                break

    def train_history(self, i, test):
        """
        Calculate mse/accuracy/loss/r2 during training. 
        Network never learns from this
        Slows down code a lot
        """
        for name, (x, t) in zip(("train", "test"), ((self.X, self.static_target), test)):
            if self.nodes[-1] > 1:
                self.history[name + "_accuracy"][i] = self.predict_accuracy(x, t)
                loss = self.predict(x) - t
                self.history[name + "_loss"][i] = max(np.mean(loss, axis=0))

            else:
                # Franke function MSE 
                pred = utils.rescale_data(self.predict(x), self.z_Franke)
                target = utils.rescale_data(t, self.z_Franke)
                self.history[name + "_mse"][i] = utils.MSE(target, pred)[0]
                self.history[name + "_R2"][i] = utils.R2(target, pred)[0]

            if test is None:
                break

    def predict(self, x):
        """
        input: x (ndarray)
        Calculate output layer with updated weights and biases
        """
        if self.converged is None:
            return np.nan

        self.Layers[0] = x
        self.feed_forward()
        return self.Layers[-1]

    def predict_accuracy(self, x, y):
        """
        Convert probabilities to accuracy score
        """
        probs = self.predict(x)
        msg = "\n\nThe probabilities do not sum to 1!\nWorry not, this probably just means there is a nan in there. Check for RuntimeWarnings in autograd.\nRerun, but with lower gamma or eta or something else\n"
        try:
            if not (abs(np.sum(probs, axis=1) - 1) < 1e-10).all():  # make sure probabilities sum to 1
                print(msg)
                return np.nan
        except:
            return np.nan
        pred = np.argmax(probs, axis=1).reshape(-1, 1)
        true = np.argmax(y, axis=1).reshape(-1, 1)
        return np.sum(pred == true) / len(true)

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
              args=args,
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
