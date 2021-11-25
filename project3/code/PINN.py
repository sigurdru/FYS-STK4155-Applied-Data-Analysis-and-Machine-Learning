"""
Physics Informed Neural Network
"""
#Import important stuff
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.candidate_sampling_ops import compute_accidental_hits
# Set random seed for reproducible results
tf.random.set_seed(0)

class PINN:
    def __init__(self, DTYPE = 'float32',
                args = 0, N_0=50, N_b=50, N_r=10000,
                tmin = 0., tmax = 1., xmin = 0., xmax = 1.,
                num_hidden_layers=8, num_neurons_per_layer=20, activation='tanh'):
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
        t_0 = tf.ones((N_0, 1), dtype=self.DTYPE)*self.lb[0]
        x_0 = tf.random.uniform((N_0, 1), self.lb[1], self.ub[1], dtype=self.DTYPE)
        X_0 = tf.concat([t_0,x_0], axis=1)

        u_0 = self.fun_u_0(t_0, x_0)
        
        # Boundary:
        t_b = tf.random.uniform((N_b, 1), self.lb[0], self.ub[0], dtype=self.DTYPE)
        x_b = self.lb[1] + (self.ub[1] - self.lb[1]) * \
            tf.keras.backend.random_bernoulli((N_b, 1), 0.5, dtype=self.DTYPE)
        # x_b = tf.random.uniform((N_b, 1), self.lb[1], self.ub[1], dtype=self.DTYPE)
        X_b = tf.concat([t_b, x_b], axis=1)

        u_b = self.fun_u_b(t_b, x_b)

        # Data:
        t_r = tf.random.uniform((N_r, 1), self.lb[0], self.ub[0], dtype=self.DTYPE)
        x_r = tf.random.uniform((N_r, 1), self.lb[1], self.ub[1], dtype=self.DTYPE)
        self.X_r = tf.concat([t_r, x_r], axis=1)
        self.X_data = [X_0, X_b]
        self.u_data = [u_0, u_b]

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
        """
        Returns initial conditions u(0,x)
        """ 
        return tf.sin(self.pi*x)

    def fun_u_b(self, t, x):
        """
        Return boundary condition
        """
        n = x.shape[0]
        return tf.zeros((n,1), dtype=self.DTYPE)

    def fun_r(self, u_t, u_xx):
        """
        return residual of the PDE
        """
        return u_t - u_xx

    def get_r(self):
        """
        Get the residual
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
        """
        Compute loss
        """
        r = self.get_r()
        phi_r = tf.reduce_mean(tf.square(r))

        loss = phi_r

        for i in range(len(self.X_data)):
            u_pred = self.model(self.X_data[i])
            loss += tf.reduce_mean(tf.square(self.u_data[i] - u_pred))

        return loss

    def get_grad(self):
        """
        Get gradient
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.compute_loss()
        
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, g
    
    @tf.function
    def train_step(self):
        # Compute current loss and gradient w.r.t. parameters
        loss, grad_theta = self.get_grad()

        # Perform gradient descent step
        self.optim.apply_gradients(zip(grad_theta, self.model.trainable_variables))

        return loss

if __name__ == '__main__':
    NN = PINN()
    N = 500
    hist = []

    for i in range(N+1):
        loss = NN.train_step()

        hist.append(loss.numpy())
        print('It {:05d}: loss = {:10.8e}'.format(i, loss))
    

    # Set up meshgrid
    N = 600
    tspace = np.linspace(NN.lb[0], NN.ub[0], N + 1)
    xspace = np.linspace(NN.lb[1], NN.ub[1], N + 1)
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(), X.flatten()]).T
    
    # Determine predictions of u(t, x)
    upred = NN.model(tf.cast(Xgrid, NN.DTYPE))
    
    # Reshape upred
    U = upred.numpy().reshape(N+1, N+1)
    
    # Surface plot of solution u(t,x)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, U, cmap='viridis')
    ax.view_init(35, 35)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_zlabel('$u_\\theta(t,x)$')
    ax.set_title('Solution of Burgers equation')
    #plt.savefig('Burgers_Solution.pdf', bbox_inches='tight', dpi=300);
    plt.show()

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.semilogy(range(len(hist)), hist, 'k-')
    ax.set_xlabel('$n_{epoch}$')
    ax.set_ylabel('$\\phi_{n_{epoch}}$')
    plt.show()
