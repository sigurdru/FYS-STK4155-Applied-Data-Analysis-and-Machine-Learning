import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dx', type=float, default=0.2)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--T', type=float, default=1.0)
parser.add_argument('--exact', type=bool, default=True)
parser.add_argument('--nr_plots', type=int, default=5)
a = parser.parse_args()

dx = a.dx
dt = a.dt
L = 1
T = a.T
exact = a.exact
nr_plots = a.nr_plots

def IC(x):
    return np.sin(np.pi*x)

u_exact = lambda x, t: np.exp(-np.pi**2*t)*np.sin(np.pi*x)
u_store = {} # To store solution for each time step

def forward_euler(L, T, IC, BC_l, BC_r, dx, dt, user_action=None):
    #dt = 0.4*dx**2 # Ensure stability
    C = dt/dx**2
    print('Stability factor:', C)

    Nx = int(round(L/dx))
    Nt = int(round(T/dt))

    x = np.linspace(0, L, Nx+1)
    t = np.linspace(0,  T, Nt+1)
    u = np.zeros(len(x)) # Solution at new time step (unknown)
    u_m = np.zeros(len(x)) # Solution at current time step (known)

    u_m = IC(x) # Initial condition
    
    if user_action is not None:
        user_action(u_m, x, t, 0) # Do something with solution..

    for n in range(Nt):
        # Interior points
        u[1:-1] = u_m[1:-1] + C*(u_m[2:] - 2*u_m[1:-1] + u_m[:-2])
        #Boundary points
        u[0] = BC_l
        u[Nx] = BC_r

        u_m[:] = u
    
        if user_action is not None:
            user_action(u_m, x, t, n+1)

    return u_m, x, t



def plot_sols(u, x, t, t_idx):
    global nr_plots, exact
    plot_idx = np.linspace(0, int(round(t[-1]/(t[1]-t[0]))), nr_plots)

    if t_idx in plot_idx: # Only plot nr_plots curves
        plt.plot(x, u, label=f't={t_idx}')
        if exact:
            plt.plot(x, u_exact(x, t[t_idx]), '--', label=f't_e={t_idx}') 

    if t_idx == len(t)-1:
        plt.legend()
        plt.show()

def print_solution(u, x, t, t_idx):
    print(f'Solution at time step {t_idx}:', u)

def store_solution(u, x, t, t_idx):
    global u_store
    u_store[t_idx] = u

def assert_local_error(u, x, t, t_idx):
    u_e = u_exact(x, t[t_idx])
    error = np.abs(u - u_e).max()
    tol = 1e-8
    assert error < tol, f'Error {error} not less than tolerance {tol}.'

def total_integrated_error(u, x, t):
    dt = t[1] - t[0]

    E = 0
    for n in range(len(t)):
        u_e = u_exact(x, t[n])
        e2 = np.sum((u[n] - u_e)**2)
        E += e2
    
    E *= dt 
    return np.round(np.sqrt(E), 3)

def test_space_steps():
    """Tests numerical scheme for time step dx=1/10
    and dx=1/100 and compares with analytical solution."""
    L = 1
    T = 0.5
    for dx in 0.1, 0.01:
        dt = 0.5*dx**2 # Satisfies stability criteria
        #u, x, t = forward_euler(L, T, IC, 0, 0, dx, dt, user_action=plot_sols)
        u, x, t = forward_euler(L, T, IC, 0, 0, dx, dt, user_action=store_solution)

        E = total_integrated_error(u_store, x, t)
        print(f'Total integrated error for dx={dx}:', E)


if __name__ == '__main__':
    BC_l = 0
    BC_r = 0

    u, x, t = forward_euler(L, T, IC, BC_l, BC_r, dx, dt, user_action=plot_sols)
    #u, x, t = forward_euler(L, T, IC, BC_l, BC_r, dx, dt, user_action=store_solution)
    test_space_steps()


