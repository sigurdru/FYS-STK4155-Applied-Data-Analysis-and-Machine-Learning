import numpy as np
import plot
import cost_activation
import matplotlib.pyplot as plt

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
        C = dt/dx**2 
    dt = args.t_step

    BC_l = args.left_boundary_condition
    BC_r = args.right_boundary_condition

    Np = args.num_plots
    test_error = args.test_error

    #defining stuff
    L = 1
    Nx = int(round(L/dx))
    Nt = int(round(T/dt))
    x = np.linspace(0, L, Nx+1)
    t = np.linspace(0, T, Nt+1)
    u = np.zeros(len(x)) # Solution at new time step (unknown)
    u_m = np.zeros(len(x)) # Solution at current time step (known)
    #u_m_final = np.zeros((len(x), Np))
    u_m_final = {}
    u_m = IC(x)
    print('Stability factor:', C)

    u_exact = lambda x, t: np.exp(-np.pi**2*t)*np.sin(np.pi*x)

    if args.study_times:
        # Study solution at two specific times 
        When_to_plot = np.array([Nt//10,Nt//2])
    else:
        # Plot Np solutions for even time periods.    
        When_to_plot = Nt//Np
        When_to_plot = np.arange(0, Nt, When_to_plot)

    for n in range(Nt):
        if n in When_to_plot:
            # Store values for plotting 
            u_m_final[n] = u_m.copy()
        
        # Interior points
        u[1:-1] = u_m[1:-1] + C * (u_m[2:] - 2*u_m[1:-1] + u_m[:-2])
        
        #Boundary points
        u[0] = BC_l
        u[Nx] = BC_r

        u_m[:] = u

    if args.test_error:
        max_error = plot.max_error_tot(x, t, u_m_final, args)
        print(f'Numerical error for dx={args.x_step}, accumulated for n={args.num_plots} time steps:', max_error)

    if args.study_times:
        plot.error_x(x, t, u_m_final, args)

    plot.Euler_solution(x, t, u_m_final, args)

    return u_m

def neural_network(args):
    """
    Solves the diffusion equation using neural natwork
    args:
        args (argparse): Information handled by the argparser 
    """
    #Importing stuff from argparse
    #Total time, x-step, t-step, left bc, right bc
    T = args.tot_time
    dx = args.x_step
    dt = args.t_step
    BC_l = args.left_boundary_condition
    BC_r = args.right_boundary_condition
    Np = args.num_plots
    #defining stuff
    #leanth of rod, num x points, num t points, x-array, y-array, initial conditions
    L = 1
    Nx = round(L/dx)
    Nt = round(T/dt)
    x = np.linspace(0, L, Nx+1)
    t = np.linspace(0, T, Nt+1)
    u_m = IC(x)


# def assert_local_error(u, x, t, t_idx):
#     u_e = u_exact(x, t[t_idx])
#     error = np.abs(u - u_e).max()
#     tol = 1e-8
#     assert error < tol, f'Error {error} not less than tolerance {tol}.'

# def total_integrated_error(u, x, t):
#     dt = t[1] - t[0]

#     E = 0
#     for n in range(len(t)):
#         u_e = u_exact(x, t[n])
#         e2 = np.sum((u[n] - u_e)**2)
#         E += e2
    
#     E *= dt 
#     return np.round(np.sqrt(E), 3)

# def test_space_steps():
#     """Tests numerical scheme for time step dx=1/10
#     and dx=1/100 and compares with analytical solution."""
#     L = 1
#     T = 0.5
#     for dx in 0.1, 0.01:
#         dt = 0.5*dx**2 # Satisfies stability criteria
#         #u, x, t = forward_euler(L, T, IC, 0, 0, dx, dt, user_action=plot_sols)
#         u, x, t = forward_euler(L, T, IC, 0, 0, dx, dt, user_action=store_solution)

#         E = total_integrated_error(u_store, x, t)
#         print(f'Total integrated error for dx={dx}:', E)


if __name__ == '__main__':
    BC_l = 0
    BC_r = 0

    u, x, t = forward_euler(L, T, IC, BC_l, BC_r, dx, dt, user_action=plot_sols)
    #u, x, t = forward_euler(L, T, IC, BC_l, BC_r, dx, dt, user_action=store_solution)
    # test_space_steps()


