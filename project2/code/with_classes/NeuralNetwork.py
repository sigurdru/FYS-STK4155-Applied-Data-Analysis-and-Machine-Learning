import numpy as np
import autograd.numpy as anp
from tqdm import tqdm
from autograd import elementwise_grad, grad

class FFNN:  # FeedForwardNeuralNetwork

    def __init__(self,
                 design,
                 target,
                 hidden_nodes=[10,],
                 batch_size=None,
                 learning_rate=0.1,
                 lmb=0,
                 activation="sigmoid",
                 cost="MSE",
                 ):

        self.X = design             # Training data
        self.t = target             # Trainging outputs
        self.N = self.X.shape[0]    # Number of input values
        self.static_target = target.copy()

        if batch_size is None:
            self.batch_size = self.N
        else:
            self.batch_size = batch_size  # Possibly redundant

        self.mini_batches = self.N // self.batch_size
        self.eta = learning_rate
        self.lmb = lmb
        bias0 = 0.1  # initial bias value

        # Array with size of nodes [21,10,10,1]
        # first corresponds to features in design matrix.
        self.nodes = np.array([self.X.shape[1], *hidden_nodes, self.t.shape[1]])


        # There are four layers (1 input, 2 hidden, 1 output)
        # Shapes:
        #  input   : (720, 21)
        #  hidden_n: (720, hidden_nodes[n])
        #  output  : (720, 1)
        self.Layers = [np.zeros((self.N, n)) for n in self.nodes]
        self.Layers[0]  = self.X.copy()

        self.z = self.Layers.copy()  # Activation layer input (z in morten's notes)
        self.delta_l = self.Layers.copy() # Activation layer input gradient (delta_l in morten's notes)

        # Initial zero for weights ensures correct association between weight nad layer indices
        self.weights = [0] + [np.random.randn(n, m) for n, m in zip(self.nodes[:-1], self.nodes[1:])]
        self.bias = [np.ones((1, n)) * bias0 for n in self.nodes]


        activation_funcs = {'sigmoid': self.sigmoid,
                            'tanh': self.tanh,
                            'relu': self.relu,
                            'leaky_relu': self.leaky_relu,
                            'softmax': self.softmax}

        cost_funcs = {'MSE': self.MSE,
                      'accuracy': self.accuracy_score}

        self.activation = activation_funcs[activation]
        self.cost = cost_funcs[cost]
        self.activation_der = elementwise_grad(self.activation)
        self.cost_der = elementwise_grad(self.cost)


    def backpropagation(self):
        """
        Updates weights and biases with backwards propagation.
         - Starts by calculating the gradients of each layer's activation function
         - Updates the weights and biases accordingly
        """
        # Calculate gradient of output layer  
        # self.delta_l[-1] = self.cost_der(self.Layers[-1]) * self.activation_der(self.z[-1])
        # self.delta_l[-1] = self.MSE_der(self.Layers[-1]) 
        self.delta_l[-1] = self.cost_der(self.Layers[-1])
        
        # Calculate gradient of previous layers backwards 
        for i in reversed(range(1, len(self.nodes) - 1)):
            self.delta_l[i] = self.delta_l[i + 1] @ self.weights[i + 1].T \
                                * self.activation_der(self.z[i])

        # Update weights and biases for each previous layer 
        # self.weights[-1] -= self.eta * self.Layers[-2].T @ self.cost(self.Layers[-1])
        for n in reversed(range(1, len(self.nodes))):
            # I don't think we should divide new weights by batch size 
            self.weights[n] -= self.eta * (self.Layers[n - 1].T @ self.delta_l[n]) #/ self.batch_size
            self.bias[n] -= self.eta * np.mean(self.delta_l[n].T, axis=1)


    def feed_forward(self):
        # Update the value at each layer from 1st hidden layer to ouput
        for n in range(1, len(self.nodes)):
            z_h = self.Layers[n - 1] @ self.weights[n] + self.bias[n]
            self.z[n] = z_h
            self.Layers[n] = self.activation(z_h)

        self.Layers[-1] = z_h

    def SGD(self):
        # Initialize randomized training data batch, and target batch
        inds = np.random.choice(np.arange(self.N), size=self.batch_size, replace=False)
        self.Layers[0] = self.X[inds]
        self.t = self.static_target[inds]

    def train(self, epochs):
        """
        Training the neural network by looping over epochs:
         1) Initializing the data [SGD]
         2) Updates the layers with weights and biases
         3) Updates weights and biases with backpropagation
        """
        indicies = np.arange(self.N)
        pbar = tqdm(range(epochs), desc="Training epochs")
        for _ in pbar:
            for i in range(self.mini_batches):
                choice = np.random.choice(indicies, size=self.batch_size, replace=False)
                self.Layers[0] = self.X[choice]
                self.t = self.static_target[choice]

                self.feed_forward()
                self.backpropagation()

    def predict(self, x):
        """
        input: x (ndarray)
        Uses the final weights and biases from the trained network
        Returns the resulting ouput layer.
        """
        self.Layers[0] = x
        self.feed_forward()
        return self.Layers[-1]

    """
    cost functions
    """
    def MSE(self, t_):
        # print('mse:')
        # print(((t_ - self.t)**2) / len(self.t))
        mse_sum = anp.sum((t_ - self.t)**2) / len(self.t)
        mse = (t_ - self.t)**2 / len(self.t)
        # print(np.shape(mse))
        # exit()
        return mse

    def accuracy_score(self, t_):
        return None


    """
    Activation functions
    """
    def sigmoid(self, x):
        return 1 / (1 + anp.exp(-x))

    def tanh(self, x):
        return anp.tanh(x)

    def relu(self, x):
        return anp.where(x > 0, x, 0)

    def leaky_relu(self, x):
        return anp.where(x > 0, x, 0.01*x)

    def softmax(self, x):
        return anp.exp(x) / anp.sum(anp.exp(x), axis=1, keepdims=True)

def MSE(y, y_):
    return sum((y - y_) ** 2) / len(y)

if __name__ == "__main__":
    # The above imports numpy as np so we have to redefine:
    # import autograd.numpy as np
    from utils import *
    from sklearn.model_selection import train_test_split

    class Args:
        num_points = 20
        epsilon = 0.2
        polynomial = 8
        dataset = "Franke"
    args = Args()
    epochs = 1000
    batch_size = int(args.num_points * args.num_points * 0.8)

    x, y, z = load_data(args)
    X = create_X(x, y, args.polynomial)

    X_, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    beta = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_train
    ols_pred = X_test @ beta

    MM = FFNN(X_,
              z_train,
              hidden_nodes=[10, 10],
              batch_size=batch_size,
              learning_rate=0.1,
              lmb=0.0,
              activation="sigmoid",
              cost="MSE")

    MM.train(epochs)
    nn_pred = MM.predict(X_test)

    print("Neural Network stochastic", MSE(z_test, nn_pred))

    print("           OLS           ", MSE(z_test, ols_pred))
