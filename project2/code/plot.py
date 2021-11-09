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
    np.random.seed()  # resetting seed
    """
    This function should automatically set filenames from args
    Also adds runtime args to a text_file for easy archivation
    """
    fname = ""
    fname += args.method + "_" + args.dataset  # reg or NN, used on dataset
    fname += "__" + func["x"] + "_" + func["y"] + "__"  # as func of
    if func["train"]:
        fname += func["train"]
    fname += "_" + func["z"]  # varying parameter
    fname += "__" + str(int(np.random.uniform() * 1e6))  # random number to identify plot
    fname += ".pdf"

    # save configuration to file
    if args.save:
        with open(archive, "a") as file:
            print("Writing run configuration to archive")
            file.write("\n\n")
            file.write(fname + "\n")
            file.write(str(args))
            file.write("\n")
            file.write(str(datetime.now()))
            file.write("\n")

    return fname

def surface_fit(data_pred, data_target, x, y, args):
    """
    Plot Franke function
    Compares fit result with data using epsilon=0.05
    """

    z_pred_ = data_pred.reshape(x.shape)
    z_target = data_target.reshape(y.shape)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x,y,z_target, alpha=0.3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf = ax.plot_surface(x,y,z_pred_, cmap=cm.jet, linewidth=0, antialiased=False)

    if args.method == "NN":
        title = r"Franke function prediction with Neural Network and noise $\epsilon=N(0,\,0.05)$"
    else:
        title = r"Franke function prediction with SGD and noise $\epsilon=N(0,\,0.05)$"

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


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
        func["train"] = name.split()[0]
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
            ylabel = r"Learning rate $\eta$"
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
            title = name + r" for Franke function, using $\eta=0.3$"
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
            ylabel = r"Learning rate $\eta$"
            xlabel = r"$\log_{10}(\lambda)$"
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
        func["train"] = name.split()[0]
        fig, ax = plt.subplots()

        for i, mse in enumerate(accuracy):
            ax.plot(mse, label=rf'{name}. $\gamma={args.gamma[i]:.2f}$')
        ax.set_title(name + "during training for different momentums")
        ax.set_xlabel("Number of epochs")
        ax.set_ylabel("MSE")
        plt.legend()
        func["x"] = "epochs"
        func["y"] = "MSE"
        func["z"] = "momentum"
        show_push_save(fig, func, args)

def eta_lambda(data, args, NN=False):
    for name, accuracy in data.items():
        print(name)
        func = defaultdict(lambda:None)
        func["train"] = name.split()[0]
        func["x"] = "eta"
        func["y"] = "lambda"
        func["z"] = name.split()[1]

        fig, ax = plt.subplots()
        col = np.round(args.lmb, 4)
        row = np.round(args.eta, 4)
        data = pd.DataFrame(accuracy, index=row, columns=col)
        sns.heatmap(data, ax=ax, annot=True, cmap=cm.coolwarm)
        if NN:
            ax.set_title(name + f" gridsearch, using {args.act_func.replace('_', ' ')} activation function on {args.dataset}-data")
        else:
            ax.set_title(name + f"as function of learning rate and lambda for {args.dataset}-data")
        ax.set_ylabel(r"$\eta$")
        ax.set_xlabel(r"$\lambda$")
        show_push_save(fig, func, args)

def train_history(NN, args):
    for mode in ("accuracy", "loss"):
        fig = plt.Figure()
        plt.plot(NN.history[f"train_{mode}"], 'o-', label="train")
        plt.plot(NN.history[f"test_{mode}"], 'o-', label="test")
        plt.legend()

        plt.title(mode + f" during training, as function of epochs")
        plt.xlabel(f"Epochs")
        plt.ylabel(mode)

        func = defaultdict(lambda: None)
        func["z"] = "train_" + mode
        func["x"] = "epochs"
        func["y"] = mode
        show_push_save(fig, func, args)