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
    plt.plot(pol_deg, MSEs, "bo--", label="test MSE")
    plt.plot(pol_deg, MSE_train, "ro--", label="Train MSE")
    plt.legend()
    plt.show()


def plot(self):
    print("Nei fuck off")
    sys.exit()
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    N = len(self.x[0])
    N = int(np.sqrt(N))
    x_ = self.x[0]  # .reshape((N, N))
    y_ = self.x[1]  # .reshape((N, N))
    x_, y_ = np.meshgrid(x_, y_)

    z_ = self.X_train @ self.test_prediction
    print(z_.shape)

    z_ = self.y.reshape((N, N))
    surf = ax.plot_surface(
        x_, y_, z_, cmap="coolwarm", lw=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()


if __name__ == "__main__":
    """
    Testing
    """
    Plot_FrankeFunction("Frank_anal")
