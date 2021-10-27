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
                 
        self.X = design             # Training data 
        self.t = target             # Trainging outputs 
        self.N = self.X.shape[0]    # Number of input values 
        self.static_target = target.copy()

        self.batch_size = self.N # Possibly redundant 
        self.eta = learning_rate
        self.lmb = lmb
        bias0 = 0.01 # initial bias value 

        # Array with size of nodes [21,10,10,1]
        # first corresponds to features in design matrix. 
        self.nodes = np.array([self.X.shape[1], *hidden_nodes, self.t.shape[1]])


        # There are four layers (1 input, 2 hidden, 1 output) 
        # Shapes:
        #  input : (720, 21)
        #  hidden: (720, 10)
        #  output: (720, 1)
        self.Layers = [np.zeros((self.N, n)) for n in self.nodes]
        self.Layers[0]  = self.X.copy()

        self.a = self.Layers.copy()  # Activation layer input (z in morten's notes)
        self.da = self.Layers.copy() # Activation layer input gradient (delta_l in morten's notes)

        # Initial zero for weights ensures correct association between weight nad layer indices
        self.weights = [0] + [np.random.randn(n, m) for n, m in zip(self.nodes[:-1], self.nodes[1:])]
        self.bias = [np.ones((1, n)) * bias0 for n in self.nodes]

        self.activation = self.sigmoid
        self.cost = self.MSE
        self.activation_der = elementwise_grad(self.activation)
        self.cost_der = elementwise_grad(self.cost)


    def update(self):
        """
        Updates weights and biases with backwards propagation.
         - Starts by calculating the gradients of each layer's activation function 
         - Updates the weights and biases accordingly 
        """
        # Calculate gradient of output layer  
        self.da[-1] = self.cost_der(self.Layers[-1]) * self.activation_der(self.a[-1])
        
        # Calculate gradient of previous layers backwards 
        for i in reversed(range(1, len(self.nodes) - 1)):
            self.da[i] = self.da[i + 1] @ self.weights[i + 1].T \
                                * self.activation_der(self.a[i])

        # Update weights and biases for each previous layer 
        for n in range(1, len(self.nodes)):
            # I don't think we should divide new weights by batch size 
            self.weights[n] -= self.eta * (self.Layers[n - 1].T @ self.da[n]) #/ self.batch_size
            self.bias[n] -= self.eta * np.mean(self.da[n].T, axis=1)

    def feed_forward(self):
        # Update the value at each layer from 1st hidden layer to ouput  
        for n in range(1, len(self.nodes)):
            z_h = self.Layers[n - 1] @ self.weights[n] + self.bias[n]
            self.a[n] = z_h
            self.Layers[n] = self.activation(z_h)

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
        pbar = tqdm(range(epochs), desc="Training epochs")
        for _ in pbar:
            self.SGD()
            self.feed_forward()
            self.update()
            pbar.update(1)

    def predict(self, x):
        """
        input: x (ndarray) 
        Uses the final weights and biases from the trained network  
        Returns the resulting ouput layer. 
        """
        self.Layers[0] = x
        self.feed_forward()
        return self.Layers[-1]

    def sigmoid(self, x):
        """
        Activation function
        """
        y = 1 / (1 + anp.exp(-x))
        return y#1 / (1 + np.exp(-x))

    def MSE(self, t_):
        """
        cost function
        """
        return (t_ - self.t) ** 2


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
