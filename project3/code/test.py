import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os

from tensorflow.python.ops.gen_array_ops import size

tf.keras.backend.set_floatx("float64")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Define trial solution, ODE rhs, loss function, and gradient method
@tf.function
def trial_solution(model, x0, t):
    #     return tf.einsum('i...,j->ij', tf.exp(-t), x0) + tf.einsum('i...,ij->ij', (1-tf.exp(-t)), model(t))
    return tf.einsum('ik,j->ij', tf.exp(-t), x0) + tf.einsum('ik,ij->ij', (1-tf.exp(-t)), model(t))


@tf.function
def rhs(model, A, x0, t):
    g = trial_solution(model, x0, t)
    return tf.einsum('ij,ij,kl,il->ik', g, g, A, g) - tf.einsum('ij,jk,ik,il->il', g, A, g, g)


@tf.function
def loss(model, A, x0, t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = trial_solution(model, x0, t)
    d_trial_dt = tape.batch_jacobian(trial, t)[:, :, 0]
    return tf.losses.MSE(d_trial_dt, rhs(model, A, x0, t))


@tf.function
def grad(model, A, x0, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, A, x0, t)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Define Rayleigh quotient


def ray_quo(A, x):
    return tf.einsum('ij,jk,ik->i', x, A, x) / tf.einsum('ij,ij->i', x, x)

# Define Euler's method


def euler_ray_quo(A, x0, T, N):
    dt = T / N
    x = [x0]
    for i in range(N - 1):
        x.append(x[-1] + dt * ((x[-1].T @ x[-1]) * A @
                               x[-1] - (x[-1].T @ A) @ x[-1] * x[-1]))

    x = np.array(x)
    ray_quo = np.einsum('ij,jk,ik->i', x, A, x) / np.einsum('ij,ij->i', x, x)

    return x, ray_quo

# Define model


class DNModel(tf.keras.Model):
    def __init__(self, n):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(50, activation=tf.nn.relu)
        self.dense_3 = tf.keras.layers.Dense(25, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(n, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.out(x)


# Define problem
# negative eigs:
# tf.random.set_seed(42)
tf.random.set_seed(222)

n = 3    # Dimension
T = 4    # Final time

# Problem formulation for Euler
N = 10000   # number of time points
# A = np.array([[3., 2., 4.], [2., 0., 2.], [4., 2., 3.]])
Q = np.random.uniform(0, 1, size=(n,n))
A = (Q.T + Q) / 2
x0 = np.random.uniform(0, 1, n)
x0 = x0 / np.linalg.norm(x0)
t = np.linspace(0, T, N)

# Problem formulation for tensorflow
Nt = 101   # number of time points
A_tf = tf.convert_to_tensor(A, dtype=tf.float64)
x0_tf = tf.convert_to_tensor(x0, dtype=tf.float64)
start = tf.constant(0, dtype=tf.float64)
stop = tf.constant(T, dtype=tf.float64)
t_tf = tf.linspace(start, stop, Nt)
t_tf = tf.reshape(t_tf, [-1, 1])

# Initial model and optimizer
model = DNModel(n)
optimizer = tf.keras.optimizers.Adam(0.005)
num_epochs = 2000

for epoch in range(num_epochs):
    cost, gradients = grad(model, A, x0_tf, t_tf)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    step = optimizer.iterations.numpy()
    if step == 1:
        print(f"Step: {step}, " + f"Loss: {tf.math.reduce_mean(cost.numpy())}")
    if step % 100 == 0:
        print(f"Step: {step}, " + f"Loss: {tf.math.reduce_mean(cost.numpy())}")

# Call models
x_euler, eig_euler = euler_ray_quo(A, x0, T, N)

s = t.reshape(-1, 1)
g = trial_solution(model, x0_tf, s)
eig_nn = ray_quo(A_tf, g)

# Print results
v, w = np.linalg.eig(A)
v_np = np.max(v)
w_np = w[:, np.argmax(v)]
if (np.sign(w_np) != np.sign(g[-1, :])).any:
    w_np *= -1
# print('A =', A)
# print('x0 =', x0)
# print('Eigvals Numpy:', v)
# print('Max Eigval Numpy', v_np)
# print('Eigvec Numpy:', w_np)
# print('Final Rayleigh Quotient Euler', eig_euler[-1])
# print('Final Rayleigh Quotient FFNN', eig_nn.numpy()[-1])
# print('Absolute Error Euler:', np.abs(np.max(v) - eig_euler[-1]))
# print('Absolute Error FFNN:', np.abs(np.max(v) - eig_nn.numpy()[-1]))
# print('Percent Error Euler', 100 *
#       np.abs((np.max(v) - eig_euler[-1]) / np.max(v)))
# print('Percent Error FFNN', 100 *
#       np.abs((np.max(v) - eig_nn.numpy()[-1]) / np.max(v)))

# Plot components of computed steady-state vector
fig0, ax0 = plt.subplots()
ax0.axhline(w_np[0], color='b', ls=':', label=f'Numpy $v_1$={w_np[0]:.5f}')
ax0.axhline(w_np[1], color='g', ls=':', label=f'Numpy $v_2$={w_np[1]:.5f}')
ax0.axhline(w_np[2], color='r', ls=':', label=f'Numpy $v_3$={w_np[2]:.5f}')
# ax0.plot(t, x_euler[:, 0], color='b', ls='--',
#          label=f'Euler $v_1$={x_euler[-1, 0]:.5f}')
# ax0.plot(t, x_euler[:, 1], color='g', ls='--',
#          label=f'Euler $v_2$={x_euler[-1, 1]:.5f}')
# ax0.plot(t, x_euler[:, 2], color='r', ls='--',
        #  label=f'Euler $v_3$={x_euler[-1, 2]:.5f}')
ax0.plot(s, g[:, 0], color='b', label=f'FFNN $v_1$={g[-1, 0]:.5f}')
ax0.plot(s, g[:, 1], color='g', label=f'FFNN $v_2$={g[-1, 1]:.5f}')
ax0.plot(s, g[:, 2], color='r', label=f'FFNN $v_3$={g[-1, 2]:.5f}')
ax0.set_ylabel('Components of vector, $v$')
ax0.set_xlabel('Time, $t$')
ax0.legend(loc='center left', bbox_to_anchor=(1.04, 0.5),
           fancybox=True, borderaxespad=0, ncol=1)

# Plot eigenvalues
fig, ax = plt.subplots()
ax.axhline(np.max(v), color='red', ls='--')
ax.plot(t, eig_euler)
ax.plot(s, eig_nn)
ax.set_xlabel('Time, $t$')
ax.set_ylabel('Rayleigh Quotient, $r$')
lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
    str(round(np.max(v), 5))
lgd_euler = "Euler $r_{\\mathrm{final}} \\sim$ " + \
    str(round(eig_euler[-1], 5))
lgd_nn = "FFNN $r_{\\mathrm{final}} \\sim$ " + \
    str(round(eig_nn.numpy()[-1], 5))
plt.legend([lgd_numpy, lgd_euler, lgd_nn], loc='center left', bbox_to_anchor=(1.04, 0.5),
           fancybox=True, borderaxespad=0, ncol=1)
plt.show()
