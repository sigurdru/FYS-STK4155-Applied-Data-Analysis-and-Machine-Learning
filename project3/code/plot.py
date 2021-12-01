"""
In this file we perform all plotting in this project.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')
here = os.path.abspath(".")
path_plots = '../output/plots/'
#archive = path_plots + "archive.txt"
def u_exact(x, t): return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


# def show_push_save(fig, func, args):
def show_save(fig, fname, args):
    """
    This function handles wether you want to show and/or save the file.

    Args:
        fig (matplotlib.figure): Figure you want to handle
        fname (string):  filename
        args (argparse)
    """
    file = path_plots + fname + '.pdf'
    if args.save:
        print(f'Saving plot: file://{here}/{file}')
        fig.savefig(file)
    if args.show:
        plt.show()
    else:
        plt.clf()
    print("\n\n")
def set_ax_info(ax, xlabel, ylabel, style='plain', title=None):
    """Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        title (str): the desired title on the axis
        xlabel (str): the desired lab on the x-axis
        ylabel (str): the desired lab on the y-axis
    """
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.ticklabel_format(style=style)
    ax.legend(fontsize=15)
def set_fname(args):
    """
    Sets fname
    """
    pass

def Euler_solution(x, t, u, args):
    fig, ax = plt.subplots()
    for n in u.keys():
        ax.plot(x, u[n], label=r'$t={}$'.format(t[n]))
    title = 'Forward-Euler Solution of 1D diffusion equation'
    xlabel = '$x$'
    ylabel = '$u(x,t)$'
    fname = 'num_sol_FE'
    set_ax_info(ax, xlabel, ylabel, title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

def max_error_tot(x, t, u, args):
    """Plot max absolute error for given time steps, and
    return the accumulated max absolute error.

    Args:
        x (array): x-coordinate
        t (array): time dimension
        u (dict): dictionary of solutions at given time step
        args: argparse arguments

    Returns:
        (int): accumulated max absolute error
    """
    fig, ax = plt.subplots()

    t_n = [t[n] for n in u.keys()]
    error_n = [np.abs(u[n] - u_exact(x, t[n])).max() for n in u.keys()]
    tot_error = np.sum(error_n)

    ax.plot(t_n, error_n, 'o--')
    
    title = 'Error between numerical and analytical solution' + '\n' 
    title += 'at different times, using $\Delta x={}$'.format(args.x_step)
    xlabel = 'Time [s]'
    ylabel = 'Max absolute error'
    fname = 'error_FE'

    set_ax_info(ax, xlabel, ylabel, style='sci', title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

    return tot_error

def error_x(x, t, u, args):
    fig, ax = plt.subplots()

    t_n = [t[n] for n in u.keys()]
    error_x = [u[n] - u_exact(x, t[n]) for n in u.keys()]

    for i, e in enumerate(error_x):
        ax.plot(x, e, '--', label=r't={}'.format(t_n[i]))
    
    title = 'Error between numerical and analytical solution' + '\n' 
    title += 'at two time levels, using $\Delta x={}$'.format(args.x_step)
    xlabel = 'x'
    ylabel = 'Absolute error'
    fname = r'error_FE_x_dx_{}'.format(args.x_step)

    set_ax_info(ax, xlabel, ylabel, style='sci', title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

def testing_data(model, args):
    """
    
    """
    t_0, x_0, u_0 = model.t_0, model.x_0, model.u_0
    t_b, x_b, u_b = model.t_b, model.x_b, model.u_b
    t_r, x_r = model.t_r, model.x_r
    fig, ax = plt.subplots()
    ax.scatter(t_0, x_0, cmap='rainbow', c=u_0, marker='X')
    ax.scatter(t_b, x_b, cmap='rainbow', c=u_b, marker='X')
    ax.scatter(t_r, x_r, c='b', marker='.', alpha=0.1)
    
    title = 'Points where we will train the network'
    xlabel = 't'
    ylabel = 'x'
    set_ax_info(ax, xlabel, ylabel, title=title)
    
    fname = 'training_points'
    show_save(fig, fname, args)

def NN_diffusion_solution(model, args):
    Nx = 20
    Nt = 2000
    tspace = np.linspace(model.lb[0], model.ub[0], Nx + 1)
    xspace = np.linspace(model.lb[1], model.ub[1], Nt + 1)
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(), X.flatten()]).T
    
    upred = model.model(Xgrid) 
    U = upred.numpy().reshape(Nx+1, Nt+1)
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # ax.contourf(T, X, U, cmap = 'hot')
    # plt.show()
    ax.plot_surface(X, T, U)
    plt.show()

def NN_diffusion_error(loss_hist, args):
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(loss_hist)

    title = 'Error of neural network during training'
    xlabel = 'iterations'
    ylabel = 'error'
    set_ax_info(ax, xlabel, ylabel, title=title)

    fname = 'NN_diffusion_error_10000'
    print(f'Error after last iteration: {loss_hist[-1]}')
    show_save(fig, fname, args)

