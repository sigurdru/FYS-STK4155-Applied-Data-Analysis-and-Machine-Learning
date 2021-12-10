import numpy as np
from tqdm import tqdm
import plot
from PINN import PINN
import matplotlib.pyplot as plt
import tensorflow as tf


def IC(x):
    """
    Returns initial conditions as a function of x
    args:
        x (array): array with x-values
    """
    return np.sin(np.pi*x)

def forward_euler(args):
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
        max_error = plot.max_error_tot(x, t, u_m_final, args)
        print(f'Numerical error for dx={args.x_step}, accumulated for n={args.num_plots} time steps:', max_error)

        all_mse = plot.MSE_FE(x, t, u, args)
        print('MSE forward euler t=0.1:', all_mse[t == 0.1])
        print('MSE forward euler t=0.5:', all_mse[t == 0.5])

    if args.study_times:
        plot.error_x(x, t, u_m_final, args)

    plot.Euler_solution(x, t, u_m_final, args)

    return u

def neural_network(args):
    """
    Solves the diffusion equation using neural natwork
    args:
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

    # plot.testing_data(NN, args)
    # plot.NN_diffusion_error(loss_hist, args)
    # plot.NN_diffusion_solution(NN, args)
    plot.NN_diffusion_error_timesteps(NN, args)


if __name__ == '__main__':
    BC_l = 0
    BC_r = 0

    # u, x, t = forward_euler(L, T, IC, BC_l, BC_r, dx, dt, user_action=plot_sols)
    #u, x, t = forward_euler(L, T, IC, BC_l, BC_r, dx, dt, user_action=store_solution)
    # test_space_steps()


