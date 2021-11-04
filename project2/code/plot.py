"""
In this file we perform all plotting in this project.
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import seaborn as sns
import pandas as pd
from datetime import datetime

# to suppress warnings from fig.tight_layout() in some plotsFalse
# import warnings
# warnings.filterwarnings("ignore")

#The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')
path_plots = '../output/plots/'
archive = path_plots + "archive.txt"


def show_push_save(fig, func, args):
    """
    This function handles wether you want to show,
    save and/or push the file.
    
    Args:
        fig (matplotlib.figure): Figure you want to handle
        func (string):  function sending fig, for easier fname creation
        args (argparse)
    """
    file = path_plots + set_fname(func, args)
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

def set_fname(func, args):
    """
    This function should automatically set filenames from args
    Also adds runtime args to a text_file for easy archivation
    """
    fname = ""
    # train or test
    if func["train"]:
        fname += "train"
    else:
        fname += "test"
    fname += "_" + func["z"]  # varying parameter
    fname += "__" + func["x"] + "_" + func["y"]  # as func of
    fname += "__" + args.method + "_" + args.dataset  # reg or NN, used on dataset
    fname += "__" + str(int(np.random.uniform() * 1e6))  # random number to identify plot
    fname += ".pdf"
    
    if args.save:
    # save configuration to file
        with open(archive, "a+") as file:
            print("Writing run configuration to archive")
            file.write("\n\n\n")
            file.write(fname + "\n")
            file.write(str(args))
            file.write("\n")
            file.write(str(datetime.now()))
            file.write("\n")

    return fname


def parameter_based(data, args):
    """
        Plot MSE heatmaps for SGD without momentum
        Different parameters are plotted based on inputs

        Number of epochs vs eta values:
         - lmb: default (float)
         - eta: array-like
         - bs : default (int)

        Number of epochs vs number of minibatches:
         - lmb: default (float)
         - eta: 0.3 (float)
         - bs : 0

        Lambda values vs eta values:
         - lmb: Array-like
         - eta: Array-like
         - bs : default (int)
    """
    for name, accuracy in data.items():
        fig, ax = plt.subplots()
        func = defaultdict(lambda: None)
        func["train"] = name.split()[0].lower()=="train"
        if len(args.lmb) == 1 and len(args.batch_size) == 1:
            """
            (x,y): (epochs, eta values)

            Good simulation:
            - Ne  : 150
            - eta : np.linspace(0.01, 0.05, 31)
            """
            cols = np.arange(args.num_epochs)
            idx = np.round(args.eta, 3)
            data = pd.DataFrame(accuracy[:,0,0,:], index=idx, columns=cols[:])
            ylabel = "Learning rate $\eta$"
            xlabel = "Number of epochs"
            title = name + " for Franke function, using 20 minibatches"
            ytick = 3
            vmax = 0.07
            
            func["x"] = "epochs"
            func["y"] = "eta"
            func["z"] = "MSE"

        if len(args.lmb) == 1 and len(args.eta) == 1:
            """
            (x,y): (epochs, number of minibatches)

            Good simulation:
            - Ne : 150
            - eta: 0.3
            - bs : 0
            """
            cols = np.arange(args.num_epochs)
            idx = args.nmb
            data = pd.DataFrame(accuracy[0,0,:,:], index=idx, columns=cols[:])
            ylabel = "Number of minibatches"
            xlabel = "Number of epochs"
            title = name + " for Franke function, using $\eta=0.3$"
            ytick = idx
            vmax = 0.07

            func["x"] = "epochs"
            func["y"] = "minibatches"
            func["z"] = "MSE"

        if len(args.lmb) != 1 and len(args.nmb) == 1:
            """
            (x,y): (log10[lambda], eta values)

            Good simulation:
            - Ne  : 200
            - lmb : np.logspace(-5, 0, 11)
            - eta : np.linspace(0.1, 0.7, 11)
            """
            cols = np.log10(args.lmb)
            idx = np.round(args.eta, 3)
            data = pd.DataFrame(accuracy[:,:,0,-1], index=idx, columns=cols[:])
            ylabel = "Learning rate $\eta$"
            xlabel = "$\log_{10}(\lambda)$"
            title = name + " for Franke function after 200 epochs, using 20 minibatches"
            ytick = idx
            vmax = None

            func["x"] = "lambda"
            func["y"] = "eta"
            func["z"] = "MSE"


        ax = sns.heatmap(data,
                        ax=ax,
                        annot=False,
                        cmap=cm.coolwarm,
                        vmax=vmax,
                        linewidths=0,
                        xticklabels=len(cols)//10,
                        yticklabels=ytick)
        ax.invert_yaxis()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        show_push_save(fig, func, args)


def momentum(data, args):
    for name, accuracy in data.items():
        func = defaultdict(lambda: None)
        func["train"] = name.split()[0].lower()=="train"
        fig, ax = plt.subplots()
        
        for i, mse in enumerate(accuracy):    
            ax.plot(mse, label=f'{name}. $\gamma={args.gamma[i]:.2f}$')
        ax.set_title(name + "during training for different momentums")
        ax.set_xlabel("Number of epochs")
        ax.set_ylabel("MSE")
        plt.legend()
        func["x"] = "epochs"
        func["y"] = "MSE"
        func["z"] = "momentum"
        show_push_save(fig, func, args)

def eta_lambda(data, args):
    pass
    