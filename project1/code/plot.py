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
    if zlabel is None:
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


def show(fig, fname, args):
    fig.savefig(os.path.join(path_plots, fname + '.pdf'))
    if args.show:
        plt.show()
    else:
        plt.clf()


def Plot_FrankeFunction(x, y, z, args):
    """Plot the Franke function and saves the plot in the output
    folder

    Args:
        x (array):  x-data
        y (array):  y-data
        z (array):  z-values
        args (argparse): argparse containing info about run
    """

    if 1 in z.shape:
        nx = x.shape[0]
        ny = x.shape[1]
        z = z.reshape((nx, ny))

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #general formalities
    fname = f"Frak_anal_eps_{args.epsilon}"
    title = 'Analytical plot of Franke\'s function'
    xlabel = '$x$'
    ylabel = '$y$'
    zlabel = '$f$'
    set_ax_info(ax, xlabel, ylabel, title, zlabel)
    fig.tight_layout()
    show(fig, fname, args)


def Plot_error(MSE_test, MSE_train, args):
    """Plot mean square error as a function of polynomial degree
    for test and train data

    Args:
        MSE_test (array): array of test mean square error
        MSE_train (array): array of train mean square error
        args (argparse): argparse containing information of method used
    """
    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(args.polynomial, MSE_test, "bo--", label="test MSE")
    ax.plot(args.polynomial, MSE_train, "ro--", label="Train MSE")

    # general formalities
    fname = 'MSE_' + args.method \
            + '_n' + str(args.num_points) \
            + '_eps' + str(args.epsilon) \
            + '_pol' + str(max(args.polynomial))
    fname = fname.replace('.','-')  # remove dots from fname
    title = 'Mean square error for ' \
            + args.method + " using " \
            + args.resampling  + " with " \
            + f"iter = {args.resampling_iter}"  
    xlabel = 'Polynomial degree'
    ylabel = 'MSE'
    set_ax_info(ax, xlabel, ylabel, title)

    # save figure
    print('Plotting error: See ' + fname + '.pdf')
    show(fig, fname, args)


def Plot_R2(R2_test, R2_train, args):
    """Plot R^2 score as a function of polynomial degree
    for test and train data

    Args:
        R2_test (array): array of test R2 score
        R2_train (array): array of train R2 score
        args (argparse): argparse containing information of method used
    """
    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(args.polynomial, R2_test, "bo--", label="Test R2")
    ax.plot(args.polynomial, R2_train, "ro--", label="Train R2")
    # general formalities
    fname = 'R2_' + args.method \
            + '_n' + str(args.num_points) \
            + '_eps' + str(args.epsilon) \
            + '_pol' + str(max(args.polynomial))
    fname = fname.replace('.', '-') # remove dots from fname
    title = 'R2 score for ' \
            + args.method
    xlabel = 'Polynomial degree'
    ylabel = 'R2'
    set_ax_info(ax, xlabel, ylabel, title)

    # save figure
    print('Plotting R2: See ' + fname + '.pdf')
    show(fig, fname, args)


def Plot_bias_var_tradeoff(datas, args):
    """Plot mean square error, variance and error as a function of polynomial degree

    Args:
        datas (defaultdict) contains following arrays
                test_errors
                test_biases
                test_vars
                train_errors
                train_biases
                train_vars
        args (argparse): argparse containing information of method used
    """
    print('Plotting Bias variance tradeoff: See output file.')
    # make figure and plot data
    fig, ax = plt.subplots()

    P = args.polynomial
    ax.plot(P, datas["test_errors"], "bo-", label="test Error")
    ax.plot(P, datas["test_biases"], "ro-", label="test Bias")
    ax.plot(P, datas["test_vars"], "go-", label="test Variance")
    ax.plot(P, datas["train_errors"], "bo--", label="train Error")
    ax.plot(P, datas["train_biases"], "ro--", label="train Bias")
    ax.plot(P, datas["train_vars"], "go--", label="train Variance")
    # General formalities
    xlabel = 'Polynomial degree'
    ylabel = 'Bias, variance and error'
    title = 'Bias Variance Tradeoff for ' + args.resampling
    set_ax_info(ax, xlabel, ylabel, title)
    # Saving figure
    fname = 'BVT_' + args.method \
            + '_n' + str(args.num_points) \
            + '_eps' + str(args.epsilon)
    show(fig, fname, args)

def Plot_BVT_lambda(results, args):
    """ Plot test MSE as function of lambda, for different polynomial degrees

    Args:
        results: (defaultdict) contains 1 2D array with test errors
                First index goes over poly-degree
                2nd index goes over lambda
    """
    fig, ax = plt.subplots()

    lmbs = args.lmb
    for i, p in enumerate(args.polynomial):
        ax.plot(np.log10(lmbs), results["test_MSE"][i], label=f"Polynomial degree: {p}")

    xlabel = "log10(lambda-parameter)"
    ylabel = "Error"
    title = f"Error as function of lambda-parameter using {args.method}"
    set_ax_info(ax, xlabel, ylabel, title)
    fname = "LBVT_" + args.method \
            + "_n" + str(args.num_points) \
            + "_eps" + str(args.epsilon)
    fname = fname.replace(".", "_|") 

    show(fig, fname, args)

if __name__ == "__main__":
    """
    Plotting analytical Franke with no noise
    """
    class Argparse:
        def __init__(self):
            self.show = True
            self.epsilon = 0

    N = 100
    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y)
    args = Argparse()
    Plot_FrankeFunction(x, y, z, args)
