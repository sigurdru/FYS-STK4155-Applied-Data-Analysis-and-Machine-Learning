"""
Physics Informed Neural Network
"""
#Import important stuff
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PINN:
    """Physics-informed neural network"""
    def __init__(self, args = None, DTYPE = 'float32',
                N_0=50, N_b=50, N_r=10000,
                tmin = 0., tmax = 1., xmin = 0., xmax = 1.,
                num_hidden_layers=8, num_neurons_per_layer=20, activation='tanh'):
        """
        Args:
            args (argparse): Information handled by the argparser
            DTYPE (str): datatype used
            N_0 (int): number of initial points
            N_b (int): number of points on boundary
            N_r (int): number of interior points
            tmin (float): start time
            tmax (float): end time
            xmin (float): start position
            xmax (float): end position
            num_hidden_layers (int): number of hidden layers
            num_neurons_per_layer (int): number of neurons per layer
            activation (str): activation function in hidden layers

        """
        # Set random seed for reproducible results
        if args:
            tf.random.set_seed(args.seed)
        else:
            tf.random.set_seed(0)
        # Set dtype
        self.DTYPE = DTYPE
        tf.keras.backend.set_floatx(self.DTYPE)

        # Constant
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
        
        # Set num datapoints
        self.N_0 = N_0
        self.N_b = N_b
        self.N_r = N_r

        # Set lower and upper bounds
        self.lb = tf.constant([tmin, xmin], dtype=self.DTYPE)
        self.ub = tf.constant([tmax, xmax], dtype=self.DTYPE)

        # Initial and boundary data
        # Iitial:
        self.t_0 = tf.ones((N_0, 1), dtype=self.DTYPE)*self.lb[0]
        self.x_0 = tf.random.uniform((N_0, 1), self.lb[1], self.ub[1], dtype=self.DTYPE)
        X_0 = tf.concat([self.t_0, self.x_0], axis=1)

        self.u_0 = self.fun_u_0(self.t_0, self.x_0)
        
        # Boundary:
        self.t_b = tf.random.uniform((N_b, 1), self.lb[0], self.ub[0], dtype=self.DTYPE)
        self.x_b = self.lb[1] + (self.ub[1] - self.lb[1]) * \
            tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype=self.DTYPE)
        X_b = tf.concat([self.t_b, self.x_b], axis=1)

        self.u_b = self.fun_u_b(self.t_b, self.x_b)

        # Data:
        self.t_r = tf.random.uniform((N_r, 1), self.lb[0], self.ub[0], dtype=self.DTYPE)
        self.x_r = tf.random.uniform((N_r, 1), self.lb[1], self.ub[1], dtype=self.DTYPE)
        self.X_r = tf.concat([self.t_r, self.x_r], axis=1)
        self.X_data = [X_0, X_b]
        self.u_data = [self.u_0, self.u_b]

        # INITIALIZE MODEL:
        # The model:
        self.model = tf.keras.Sequential()

        # Input
        self.model.add(tf.keras.Input(2))
        
        # Map input to lb and ub:
        scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        )
        self.model.add(scaling_layer)

        # Hidden layers:
        for _ in range(num_hidden_layers):
            self.model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                            activation=tf.keras.activations.get(activation),
                                            kernel_initializer='glorot_normal'))
        
        # Output layer
        self.model.add(tf.keras.layers.Dense(1))

        # Set optimizer
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [1000, 3000], [1e-2, 1e-3, 5e-4]
        )

        self.optim = tf.keras.optimizers.Adam(learning_rate=lr)

    def fun_u_0(self, t, x):
        """Returns initial conditions u(0,x).

        Args:
            t (array): time domain
            x (array): spatial domain

        """ 
        return tf.sin(self.pi*x)

    def fun_u_b(self, t, x):
        """Returns boundary conditions.

        Args:
            t (array): time domain
            x (array): spatial domain

        """ 
        n = x.shape[0]
        return tf.zeros((n,1), dtype=self.DTYPE)

    def fun_r(self, u_t, u_xx):
        """Returns residual of the PDE.

        Args:
            u_t (array): first order time derivative of solution
            u_xx (array): second order spatial derivative of solution

        """
        return u_t - u_xx

    def get_r(self):
        """Return residual of numerical solution.

        """
        with tf.GradientTape(persistent=True) as tape:
            t, x = self.X_r[:, 0:1], self.X_r[:, 1:2]

            tape.watch(t)
            tape.watch(x)

            u = self.model(tf.stack([t[:, 0], x[:, 0]], axis=1))
            u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)

        del tape

        return self.fun_r(u_t, u_xx)

    def compute_loss(self):
        """Compute loss using mean squared loss function.

        """
        r = self.get_r()
        phi_r = tf.reduce_mean(tf.square(r))

        loss = phi_r

        for i in range(len(self.X_data)):
            u_pred = self.model(self.X_data[i])
            loss += tf.reduce_mean(tf.square(self.u_data[i] - u_pred))

        return loss

    def get_grad(self):
        """Cacluate gradient of loss function.

        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.compute_loss()
        
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape

        return loss, g
    
    @tf.function
    def train_step(self):
        """Run one training step and update weights and biases.
        """
        # Compute current loss and gradient w.r.t. parameters
        loss, grad_theta = self.get_grad()

        # Perform gradient descent step
        self.optim.apply_gradients(zip(grad_theta, self.model.trainable_variables))

        return loss