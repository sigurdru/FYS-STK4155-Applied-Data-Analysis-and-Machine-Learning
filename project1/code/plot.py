import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os, re

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from random import random, seed
import utils

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
    z = utils.FrankeFunction(x, y)
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

def Plot_error(MSE_test, MSE_train, args):
    """Plot mean square error as a function of polynomial degree
    for test and train data

    Args:
        MSE_test (array): array of test mean square error
        MSE_train (array): array of train mean square error
        args (argparse): argparse containing information of method used
    """
    #Plot the data
    fig, ax = plt.subplots()
    ax.plot(args.polynomial, MSE_test, "bo--", label="test MSE")
    ax.plot(args.polynomial, MSE_train, "ro--", label="Train MSE")
    #general formalities
    fname = 'MSE_' + args.method \
            + '_n' + str(args.num_points) \
            + '_eps' +str(args.epsilon) \
            + '_pol' +str(max(args.polynomial))
    #So we dontget any dots in the fnames
    fname = fname.replace('.','-')
    title = 'Mean square error for ' \
            + args.method
    xlabel = 'Polynomial degree'
    ylabel = 'MSE'
    set_ax_info(ax, xlabel, ylabel, title)
    #save figure
    fig.savefig(os.path.join(path_plots, fname + '.pdf'))
    print('Plotting error: See '+ fname + '.pdf')
    plt.close()

def Plot_R2(R2_test, R2_train, args):
    """Plot R^2 score as a function of polynomial degree
    for test and train data

    Args:
        R2_test (array): array of test R2 score
        R2_train (array): array of train R2 score
        args (argparse): argparse containing information of method used
    """
    #Plot the data
    fig, ax = plt.subplots()
    ax.plot(args.polynomial, R2_test, "bo--", label="Test R2")
    ax.plot(args.polynomial, R2_train, "ro--", label="Train R2")
    #general formalities
    fname = 'R2_' + args.method \
            + '_n' + str(args.num_points) \
            + '_eps' + str(args.epsilon) \
            + '_pol' + str(max(args.polynomial))
    #So we dontget any dots in the fnames
    fname = fname.replace('.', '-')
    title = 'R2 score for ' \
            + args.method
    xlabel = 'Polynomial degree'
    ylabel = 'R2'
    set_ax_info(ax, xlabel, ylabel, title)
    #save figure
    fig.savefig(os.path.join(path_plots, fname + '.pdf'))
    print('Plotting R2: See ' + fname + '.pdf')
    plt.close()

def Plot_bias_var_tradeoff(test_errors, test_biases, test_vars, train_errors,
                            train_biases, train_vars, args):
    """Plot mean square error, variance and error as a function of polynomial degree

    Args:
        test_errors  (array) array containing test error
        test_biases  (array) array containing test bias
        test_vars    (array) array containing test variance
        train_errors (array) array containing train error
        train_biases (array) array containing train bias
        train_vars   (array) array containing train variance
        args (argparse): argparse containing information of method used
    """
    print('Plotting Bias variance tradeoff: See output file.')
    #make figure and plot data
    fig, ax = plt.subplots()
    
    P = args.polynomial
    ax.plot(P, test_errors, "bo-", label="test Error")
    ax.plot(P, test_biases, "ro-", label="test Bias")
    ax.plot(P, test_vars, "go-", label="test Variance")
    ax.plot(P, train_errors, "bo--", label="train Error")
    ax.plot(P, train_biases, "ro--", label="train Bias")
    ax.plot(P, train_vars, "go--", label="train Variance")
    #General formalities
    xlabel = 'Polynomial degree'
    ylabel = 'Bias, variance and error'
    title = 'Bias Variance Tradeoff for ' + args.resampling
    set_ax_info(ax, xlabel, ylabel, title)
    #Saving figure
    fname = fname = 'BVT_' + args.method \
            + '_n' + str(args.num_points) \
            + '_eps' + str(args.epsilon) 
    fig.savefig(os.path.join(path_plots, fname + '.pdf'))
    plt.close()

if __name__ == "__main__":
    """
    Testing
    """
    Plot_FrankeFunction("Frank_anal")
