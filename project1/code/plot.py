import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os, re

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from random import random, seed
import ord_lstsq

# to suppress warnings from fig.tight_layout() in some plotsFalse
# import warnings
# warnings.filterwarnings("ignore")

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')
path_plots = '../output/plots'


def set_ax_info(ax, xlabel, ylabel, title=None, zlabel=None):
    """Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        title (str): the desired title on the axis
        xlabel (str): the desired lab on the x-axis
        ylabel (str): the desired lab on the y-axis
        zlabel (str): the desired lab on the z-axis 

    """
    if zlabel == None:
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.ticklabel_format(style='plain')
        ax.legend(fontsize=15)
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_zlabel(zlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)

def Plot_FrankeFunction(fname):
    """Plot the Franke function and saves the plot in the output
    folder

    Args:
        fname (str): name of the output file
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = ord_lstsq.FrankeFunction(x, y)
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #general formalities
    title = 'Analytical plot of Franke\'s function'
    xlabel = '$x$'
    ylabel = '$y$'
    zlabel = '$f$'
    set_ax_info(ax, xlabel, ylabel, title, zlabel)
    fig.tight_layout()
    fig.savefig(os.path.join(path_plots, fname + '.pdf'))
    plt.close()

def Plot_error(pol_deg, MSE_test, MSE_train, args):
    """Plot mean square error as a function of polynomial degree
    for test and train data

    Args:
        pol_deg (array): array of polynomial degree
        MSE_test (array): array of test mean square error
        MSE_train (array): array of train mean square error
        args (argparse): argparse containing information of method used
    """
    #Plot the data
    fig, ax = plt.subplots()
    ax.plot(pol_deg, MSE_test, "bo--", label="test MSE")
    ax.plot(pol_deg, MSE_train, "ro--", label="Train MSE")
    #general formalities
    fname = 'MSE_' + args.method \
            + '_n' + str(args.num_points) \
            + '_eps' +str(args.epsilon) \
            + '_pol' +str(max(args.polynomial)) 
    title = 'Mean square error for ' \
            + args.method
    xlabel = 'Polynomial degree'
    ylabel = 'Error'
    set_ax_info(ax, xlabel, ylabel, title)
    #save figure
    fig.savefig(os.path.join(path_plots, fname + '.pdf'))
    plt.close()


def Plot_error(args):
    """
    Up next: plot error, variance and bias
    """
    pass

if __name__ == "__main__":
    """
    Testing
    """
    Plot_FrankeFunction("Frank_anal")
