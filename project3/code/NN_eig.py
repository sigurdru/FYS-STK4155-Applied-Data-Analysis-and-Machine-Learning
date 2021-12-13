import numpy as np
import tensorflow as tf
import os

tf.keras.backend.set_floatx("float64")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Define trial solution, ODE rhs, loss function, and gradient method
@tf.function
def trial_solution(model, x0, t):
    """Trial solution of NN, satisfying initial condition x(0) = x_0."""
    #     return tf.einsum('i...,j->ij', tf.exp(-t), x0) + tf.einsum('i...,ij->ij', (1-tf.exp(-t)), model(t))
    return tf.einsum('ik,j->ij', tf.exp(-t), x0) + tf.einsum('ik,ij->ij', (1-tf.exp(-t)), model(t))


@tf.function
def rhs(model, A, x0, t):
    """Right hand side of ODE."""
    g = trial_solution(model, x0, t)
    return tf.einsum('ij,ij,kl,il->ik', g, g, A, g) - tf.einsum('ij,jk,ik,il->il', g, A, g, g)


@tf.function
def loss(model, A, x0, t):
    """MSE of difference between rhs and lhs"""
    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = trial_solution(model, x0, t)
    d_trial_dt = tape.batch_jacobian(trial, t)[:, :, 0]
    return tf.losses.MSE(d_trial_dt, rhs(model, A, x0, t))


@tf.function
def grad(model, A, x0, t):
    """Calculates gradient of loss function. To be minimized."""
    with tf.GradientTape() as tape:
        loss_value = loss(model, A, x0, t)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def ray_quo(A, x):
    """Rayleigh quotient. Returns eigenvalue corresponding to
    an eigenvector x of A."""
    return tf.einsum('ij,jk,ik->i', x, A, x) / tf.einsum('ij,ij->i', x, x)


class DNModel(tf.keras.Model):
    """Deep neural network model to find eigenvalues."""
    def __init__(self, n):
        """Initialize three hidden layers + an output layer"""
        super(DNModel, self).__init__() # Import parameters from keras model

        self.dense_1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(50, activation=tf.nn.relu)
        self.dense_3 = tf.keras.layers.Dense(25, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(n, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.out(x)


def euler_ray_quo(A, x0, T, N):
    """Rayleigh quotient from Forward Euler scheme.
    Returns:
        x (array): Eigenvectors corresponding to largest eigenvalue
        ray_quo (array): Estimated largest eigenvalue for each time step"""

    t = np.linspace(0, T, N)
    dt = t[1] - t[0]
    x = {}
    x[0] = x0

    for n in range(N-1):
        x[n+1] = x[n] + dt*(x[n].T @ x[n] * A @ x[n]) - dt*(x[n].T @ A @ x[n] * x[n])
    
    x = np.array(list(x.values())) # dict -> array
    ray_quo = np.einsum('ji,jk,ik->i', x.T, A, x) / np.einsum('ji,ij->i', x.T, x)

    return x, ray_quo