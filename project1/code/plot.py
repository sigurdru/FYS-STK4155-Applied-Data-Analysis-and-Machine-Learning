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
    fname = f"Frank_anal_eps_{args.epsilon}"
    fname = fname.replace('.', '')  # remove dots from fname
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
    fname = fname.replace('.','')  # remove dots from fname
    title = 'Mean square error for ' + args.method
    if args.resampling != "None":
        title += " using " \
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
    fname = fname.replace('.', '') # remove dots from fname
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
    fname = fname.replace(".", "")

    show(fig, fname, args)


def Plot_VarOLS(args):
    """Here we plot the parameters for different polynomials
    with confidence intervals.
    """
    p = np.array([1, 3, 5])
    n = 100
    sigma_sq = args.epsilon
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x,y = np.meshgrid(x,y)
    z = utils.FrankeFunction(x, y, sigma_sq)
    for i in p:
        X = utils.create_X(x, y, i)
        XTXinverse = np.linalg.pinv(X.T @ X)
        beta = XTXinverse @ X.T @ z
        variance_beta = sigma_sq*XTXinverse
        sigma_sq_beta = np.diag(variance_beta)
        upper = beta[:,0] + 2*np.sqrt(sigma_sq_beta)
        lower = beta[:,0] - 2*np.sqrt(sigma_sq_beta)
        beta_index = np.arange(0,len(beta), 1)
        fig, ax = plt.subplots()
        ax.scatter(beta_index, beta, label = r'$\beta$(index)')
        ax.fill_between(beta_index, lower, upper, alpha = 0.2, label = r'$\pm 2\sigma$')
        xlabel = r'index'
        ylabel = r'$\beta$'
        title = r'$\beta$-values with confidence interval for p = %i' %(i)
        set_ax_info(ax, xlabel, ylabel, title)
        fname = 'Var_OLS_poldeg_' + str(i)
        show(fig,fname, args)
        print('Plotting variance in beta: See ' + fname + '.pdf')

if __name__ == "__main__":
    """
    Here we can plot Franke function with or withot noise,
    simply change the self.epsilon value to the desired noise.
    Then run python3 plot.py, and it will generate a file with the appropriate
    name.
    """
    class Argparse:
        def __init__(self, eps = 0):
            self.show = True
            self.epsilon = eps

    N = 100
    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)
    args = Argparse()
    z = utils.FrankeFunction(x, y, eps=args.epsilon)
    Plot_FrankeFunction(x, y, z, args)
    # Plot_VarOLS()
