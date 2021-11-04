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
path_plots = '../output/plots/'


def show_push_save(fig, args):
    """
    This function handles wether you want to show,
    save and/or push the file.
    
    Args:
        fname (str): Name of file
        fig (matplotlib.figure): Figure you want to handle
        args (argparse)
    """
    file = path_plots + set_fname(args) + ".pdf"
    if args.save:
        print("Saving plot: ", file)
        fig.savefig(file)
    if args.push:
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