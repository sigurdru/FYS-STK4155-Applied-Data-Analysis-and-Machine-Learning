"""
In this file we perform all plotting in this project.
"""
import matplotlib.pyplot as plt
import os


#The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')
here = os.path.abspath(".")
path_plots = '../output/plots/'
archive = path_plots + "archive.txt"


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

def set_fname(args):
    """
    Sets fname
    """
    pass

def Euler_solution(x, y, args):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fname = 'some_fname'
    show_save(fig, fname, args)


