"""
In this file we perform all plotting in this project.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# to suppress warnings from fig.tight_layout() in some plotsFalse
# import warnings
# warnings.filterwarnings("ignore")

#The style we want
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

def show_push_save(fig, fname, args):
    """
    This function handles wether you want to show,
    save and/or push the file.
    
    Args:
        fname (str): Name of file
        fig (matplotlib.figure): Figure you want to handle
        args (argparse)
    """
    if args.save:
        print("Saving plot: ", fname + ".pdf")
        fig.savefig(os.path.join(path_plots, fname + '.pdf'))
    if args.push:
        file = os.path.join(path_plots, fname + ".pdf")
        os.system(f"git add {file}")
        os.system("git commit -m 'plots'")
        os.system("git push")
        print(f"Pushed to git: {file}")
    if args.show:
        plt.show()
    else:
        plt.clf()

def set_fname(args):
    """
    This function should automatically set filenames so we don't
    get redundant code. Add functionality when needed.
    """
    pass