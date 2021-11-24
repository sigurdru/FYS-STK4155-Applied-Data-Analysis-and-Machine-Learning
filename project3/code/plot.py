"""
In this file we perform all plotting in this project.
"""
import matplotlib.pyplot as plt
import numpy as np
import os


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
def set_ax_info(ax, xlabel, ylabel, title=None):
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
    ax.ticklabel_format(style='plain')
    ax.legend(fontsize=15)
def set_fname(args):
    """
    Sets fname
    """
    pass

def Euler_solution(x, t, u, args):
    fig, ax = plt.subplots()
    for n in u.keys():
        ax.plot(x, u[n], label=r'$\chi_{n}$')
    title = 'Numerical Solution of Euler-forward'
    xlabel = 'x'
    ylabel = 'y'
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
    tot_error = 0
    fig, ax = plt.subplots()

    for n in u.keys():
        error_n = np.abs(u[n] - u_exact(x, t[n])).max()
        ax.plot(n, error_n, label=f't={n}')
        tot_error += error_n 

    title = 'Absolute error between numerical and analytical solution'
    xlabel = 'time'
    ylabel = 'error'
    fname = 'error_FE'
    set_ax_info(ax, xlabel, ylabel, title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

    return tot_error

