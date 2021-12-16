import numpy as np
from tqdm import tqdm
import plot
from PINN import PINN
import NN_eig
import tensorflow as tf


def IC(x):
    """
    Returns initial conditions as a function of x
    args:
        x (array): array with x-values
    """
    return np.sin(np.pi*x)

def forward_euler(args):
    """Forward euler method to solve 1D diffusion equation.
    
    Args:
        args (argparse): Information handled by the argparser
        
    Returns:
        u (array): numerical solution for all time steps

    """
    #Importing stuff from argparse
    T = args.tot_time
    dx = args.x_step

    if args.t_step == 0:
        C = args.stability_criterion 
        args.t_step = C * dx**2 
    else:
        C = args.t_step/dx**2 
    dt = args.t_step

    BC_l = args.left_boundary_condition
    BC_r = args.right_boundary_condition

    Np = args.num_plots

    #defining stuff
    L = 1
    Nx = int(round(L/dx))
    Nt = int(round(T/dt))
    x = np.linspace(0, L, Nx+1)
    t = np.linspace(0, T, Nt+1)
    u = np.zeros((len(t), len(x)))
    u_m_final = {}
    
    u[0,:] = IC(x)
    print('Stability factor:', C)

    u_exact = lambda x, t: np.exp(-np.pi**2*t)*np.sin(np.pi*x)

    if args.study_times:
        # Study solution at two specific times 
        When_to_plot = np.array([Nt//10,Nt//2])
    else:
        # Plot Np solutions for even time periods.    
        When_to_plot = Nt//Np
        When_to_plot = np.arange(0, Nt, When_to_plot)

    pbar = tqdm(range(Nt), desc = 'Training progressions')
    
    for n in pbar:
        if n in When_to_plot:
            # Store values for plotting 
            u_m_final[n] = u[n,:].copy()
        
        # Interior points
        u[n+1, 1:-1] = u[n, 1:-1] + C * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        
        #Boundary points
        u[n+1, 0] = BC_l
        u[n+1, Nx] = BC_r

    if args.test_error:
        all_mse = plot.MSE_FE(x, t, u, args)
        print('MSE forward euler t=0.1:', all_mse[t == 0.1])
        print('MSE forward euler t=0.5:', all_mse[t == 0.5])

    if args.study_times:
        plot.error_x(x, t, u_m_final, args)

    plot.Euler_solution(x, t, u_m_final, args)

    return u

def neural_network(args):
    """
    Solves the diffusion equation using neural natwork.

    Args:
        args (argparse): Information handled by the argparser 

    """
    # Setup of Neural Network
    #set default values
    tmin = 0.
    xmin = 0.
    xmax = 1.

    # Import stuff from argparse
    tmax = args.tot_time
    DTYPE = args.datatype
    N_0 = args.num_initial_points
    N_b = args.num_boundary_points
    N_r = args.num_train_points
    num_hidden_layers = args.num_hidden_layers
    num_neurons_per_layer = args.num_neurons_per_layer
    activation = args.activation_function

    NN = PINN(args = args, DTYPE=DTYPE,
            N_0 = N_0, N_b = N_b, N_r = N_r,
            tmin = tmin, tmax = tmax, xmin = xmin, xmax = xmax,
            num_hidden_layers = num_hidden_layers, num_neurons_per_layer = num_neurons_per_layer,
            activation = activation)
    NN.model.summary()
    
    N = args.num_train_iter
    # Training
    loss_hist = []
    loss = 0
    pbar = tqdm(range(N + 1), desc = 'Training progressions')#, desc=f"eta: {loss:.6f}, lambda: {lmb:.6f}. Training")
    for _ in pbar:
        loss = NN.train_step()
        loss_hist.append(loss.numpy())

    plot.testing_data(NN, args)
    plot.NN_diffusion_error(loss_hist, args)
    plot.NN_diffusion_solution(NN, args)
    plot.NN_diffusion_error_timesteps(NN, args)

def neural_network_eig(args):
    """
    Finds eigenvalues and vectors with neural network.

    Args:
        args (argparse): Information handled by the argparser 

    """
    n = args.dimension    # Dimension
    T = args.tot_time     # Final time
    
    N = args.N_t_points   # number of time points (FE)
    Q = np.random.uniform(0, 1, size=(n,n))
    A = (Q.T + Q) / 2
    x0 = np.random.uniform(0, 1, n)
    x0 = x0 / np.linalg.norm(x0)
    t = np.linspace(0, T, N)
    
    # Problem formulation for tensorflow
    Nt = args.N_t_points   # number of time points (NN)
    A_tf = tf.convert_to_tensor(A, dtype=tf.float64)
    x0_tf = tf.convert_to_tensor(x0, dtype=tf.float64)
    start = tf.constant(0, dtype=tf.float64)
    stop = tf.constant(T, dtype=tf.float64)
    t_tf = tf.linspace(start, stop, Nt)
    t_tf = tf.reshape(t_tf, [-1, 1])
    
    # Initial model and optimizer
    model = NN_eig.DNModel(n)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate) # 0.005
    num_epochs = args.nr_epochs # 2000
    
    for epoch in range(num_epochs):
        cost, gradients = NN_eig.grad(model, A, x0_tf, t_tf)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
        step = optimizer.iterations.numpy()
        if step == 1:
            print(f"Step: {step}, " + f"Loss: {tf.math.reduce_mean(cost.numpy())}")
        if step % 100 == 0:
            print(f"Step: {step}, " + f"Loss: {tf.math.reduce_mean(cost.numpy())}")
    
    # Call models
    s = t.reshape(-1, 1)
    g = NN_eig.trial_solution(model, x0_tf, s)
    eigval_nn = NN_eig.ray_quo(A_tf, g)
    eigvec_fe, eigval_fe = NN_eig.euler_ray_quo(A, x0, T, N)

    v, w = np.linalg.eig(A)
    v_np = np.max(v)
    w_np = w[:, np.argmax(v)]
    w_np = np.where(np.sign(w_np) == np.sign(g[-1,:]), w_np, -w_np)

    print('A =', A)
    print('x0 =', x0)
    print('Max Eigval Numpy', v_np)
    print('Eigvec Numpy:', w_np)
    print('Final Rayleigh Quotient NN', eigval_nn.numpy()[-1])
    print('Final Rayleigh Quotient FE', eigval_fe[-1])
    print('Relative Error NN', 100 *
          np.abs((np.max(v) - eigval_nn.numpy()[-1]) / np.max(v)))
    print('Relative Error FE', 100 *
          np.abs((np.max(v) - eigval_fe[-1]) / np.max(v)))

    if args.dimension == 3:
        plot.plot_eig_dim3(w_np, g, eigvec_fe, eigval_nn, eigval_fe, s, t, v, args)
    elif args.dimension == 6:
        plot.plot_eig_dim6(w_np, g, eigvec_fe, eigval_nn, eigval_fe, s, t, v, args)

