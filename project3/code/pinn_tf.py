import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time

from tensorflow.python.eager.backprop import GradientTape
tf.keras.backend.set_floatx("float64")

def g0(t, x0):
    """Initial function satisfying initial condition in trial solution.
    t (array): independent time variable
    x0 (array): initial condition
    """
    print(np.shape(x0))
    print(np.shape(t))
    print(np.shape(tf.einsum('ik,jl->ij', tf.exp(-t), x0)))
    exit()
    return tf.einsum('ik,jl->ij', tf.exp(-t), x0)

def gNN(t, NN_model):
    """Part of trial solution given by NN model.
    NB: Must be zero for initial solution!
    t (array): independent time variable
    NN_model: tf neural network model
    """
    #return tf.einsum('ik,lj->ij', t, NN_model(t)) # Satisfies initial condition
    return tf.einsum('ik,lj->ij', 1-tf.exp(-t), NN_model(t)) # Satisfies initial condition

@tf.function
def trial_solution(t, x0, NN_model):
    """Define trial solution -> g0(t) + X*NN(x,P).
    t (array): independent time variable
    g0 (ndarray): initial condition g0(t)
    NN_model: tf neural network model
    """
    return g0(t, x0) + gNN(t, NN_model)

@tf.function
def RHS(t, x0, A, NN_model):
    """RHS of differential equation to solve"""
    g = trial_solution(t, x0, NN_model)
    return tf.einsum('ij,ij,kl,il->ik', g, g, A, g) - tf.einsum('ij,jk,ik,il->il', g, A, g, g)

@tf.autograph.experimental.do_not_convert
def loss(t, x0, A, NN_model):
    """Loss function. Approximate dg/dt (LHS) and calculate 
    residual (MSE) in differential equation (RHS - LHS).
    RHS known. LHS approximated.
    """
    with tf.GradientTape() as tape:
        tape.watch(t)
        g = trial_solution(t, x0, NN_model)

    dg_dt = tape.batch_jacobian(g, t)[:, :, 0]

    return tf.losses.MSE(dg_dt, RHS(t, x0, A, NN_model))

@tf.function
def grad(t, x0, A, NN_model):
    """Calculate gradient of loss function. Result defines equilibrium points."""
    with GradientTape() as tape:
        loss_val = loss(t, x0, A, NN_model)

    loss_grad = tape.gradient(loss_val, NN_model.trainable_variables)

    return loss_val, loss_grad

def rayleigh_quotient(x, A):
    """Formula for eigenvalue of A corresponding to eigenvector x of A."""
    return tf.einsum('ij,jk,ik->i', x, A, x) / tf.einsum('ij,ij->i', x, x)

class DE_NeuralNetwork(tf.keras.Model):
    """Neural network model for solving specific differential equation
    and estimating eigenvalues of given matrix A."""
    def __init__(self, n):
        """n: number of output nodes, i.e. nr dimensions"""
        super(DE_NeuralNetwork, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(50, activation=tf.nn.sigmoid)
        self.dense_3 = tf.keras.layers.Dense(25, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(n, name="output")

    def __call__(self, inputs):
        x1 = self.dense_1(inputs)
        x2 = self.dense_2(x1)
        x3 = self.dense_3(x2)
        return self.out(x3)

if __name__ == '__main__':
    n = 3 # dimensions (nr outputs => nr eigeinvalues)
    T0, T0_tf = 0, tf.constant(0)
    T1, T1_tf = 1, tf.constant(1) # end time
    N, N_tf = 1001, tf.constant(1001, dtype=tf.int64) # nr time steps

    t = np.linspace(T0, T1, N).reshape(-1,1) # independent time variable (np)
    t_tf = tf.linspace(T0_tf, T1_tf, N_tf) # independent time variable (tf)
    t_tf = tf.reshape(t_tf, [-1, 1]) # Convert to tf column vector
    
    Q = np.random.randint(4, size=n**2).reshape(n,n)
    A = np.asmatrix((Q.T + Q)/2) # Make symmetric matrix
    A_tf = tf.convert_to_tensor(A, dtype=tf.float64)

    x0 = np.random.uniform(0, 1, size=n) # intial condition - NB: must be same dimension as NN_model (nr outputs)
    x0 = np.array(x0) / np.linalg.norm(x0)
    x0 = tf.convert_to_tensor(x0, dtype=tf.float64)
    x0 = tf.reshape(x0, [-1,1]) #shape (n,1)

    # g0_test = g0(t, x0)
    # print('g0', g0_test.shape)

    NN_model = DE_NeuralNetwork(n)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # epochs = 2000
    epochs = 4

    print(NN_model(t_tf).shape)

    gNN_test = gNN(t, NN_model)
    print('gN', gNN_test.shape)
    # Train neural network
    for e in range(epochs):
        loss_val, loss_grad = grad(t_tf, x0, A_tf, NN_model)
        optimizer.apply_gradients(zip(loss_grad, NN_model.trainable_variables)) # use adam gradient descent on each trainable parameter
        step = optimizer.iterations.numpy()

        if step % 100 == 0:
            print(f'Step {step}, Loss: {tf.math.reduce_mean(loss_val.numpy())}')
        
    g = trial_solution(t, x0, NN_model)
    eigvals_predict = rayleigh_quotient(g, A_tf)

    eigvals_true, eigvec_true = np.linalg.eig(A)
    print(np.shape(eigvals_predict))
    print(np.shape(eigvals_true))
    print('Predicted eigvals:', eigvals_predict.numpy())
    print('True eigenvalues:', eigvals_true)




