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
