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


#The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')
here = os.path.abspath(".")
path_plots = '../output/plots/'
archive = path_plots + "archive.txt"


def show_push_save(fig, func, args):
    """
    This function handles wether you want to show,
    save and/or push the file to git.

    Args:
        fig (matplotlib.figure): Figure you want to handle
        func (string):  function sending fig, for easier fname creation
        args (argparse)
    """
    file = path_plots + set_fname(func, args)
    if args.save:
        print(f'Saving plot: file://{here}/{file}')
        # print("Saving plot: ", file)
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
    print("\n\n")

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
    if args.act_func != "sigmoid":
        fname += "__" + args.act_func
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
        title = "SGD prediction of Franke function with noise $\epsilon=N(0,\,0.05)$.\n" \
                + "Using $\eta={:.2f}$, $\gamma={:.1f}$ and $10^{:.0f}$ epochs".format(args.eta[0], args.gamma, np.log10(args.num_epochs))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title, fontsize=15)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    # exit()


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
            - eta : np.linspace(0.01, 0.5, 21)
            """
            cols = np.arange(args.num_epochs)
            idx = np.round(args.eta, 3)
            data = pd.DataFrame(accuracy[:,0,0,:], index=idx, columns=cols[:])
            
            title = name + " for Franke function\n" + "using 20 minibatches"
            xlabel = "Number of epochs"
            
            if args.dynamic_eta:
                ylabel = r"Initial learning rate $\eta_0$"
                title += r" and dynamic $\eta$"
                func["y"] = "dynamic_eta"
            
            else:
                ylabel = r"Learning rate $\eta$"
                func["y"] = "eta"

            xtick = len(cols)//10
            xrot=0

            vmax = 0.07

            func["x"] = "epochs"
            func["z"] = "MSE"

        if len(args.lmb) == 1 and len(args.eta) == 1:
            """
            (x,y): (epochs, number of minibatches)

            Good simulation:
            - Ne : 150
            - eta: 0.25
            - bs : 0
            """
            cols = np.arange(args.num_epochs)
            idx = args.nmb
            data = pd.DataFrame(accuracy[0,0,:,:], index=idx, columns=cols[:])
            ylabel = "Number of minibatches"
            xlabel = "Number of epochs"
            title = name + r" for Franke function, using $\eta=0.25$"
            ytick = idx
            xtick = len(cols) // 10
            vmax = 0.07
            xrot=0

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
            title = name + " for Franke function after 150 epochs\n" + "using 20 minibatches"
            ytick = idx
            xtick = cols
            xrot=0
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
                        xticklabels=xtick,
                        yticklabels=idx)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, fontsize=12)
        ax.invert_yaxis()
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_title(title, fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)
        show_push_save(fig, func, args)


def momentum(data, args):
    """
    Plots MSE as function of eta and epochs, for runs with different momentum factors

    Optimal simulation:
     - Ne  : 100
     - eta : 0.25
    """
    for name, accuracy in data.items():
        func = defaultdict(lambda: None)
        func["train"] = name.split()[0]
        fig, ax = plt.subplots()

        cols = np.arange(args.num_epochs)
        idx = np.round(args.gamma,2)
        data = pd.DataFrame(accuracy, index=idx, columns=cols)
        ylabel = r"Momentum parameter $\gamma$"
        xlabel = "Number of epochs"
        title = name + r" for Franke function, using $\eta=0.25$" + "\n" + "and 20 minibatches"
        xtick = len(cols)//10
        ytick = idx
        xrot = 0

        func["x"] = "epochs"
        func["y"] = "gamma"
        func["z"] = "MSE"

        ax = sns.heatmap(data,
                        ax=ax,
                        annot=False,
                        cmap=cm.coolwarm,
                        vmax=0.07,
                        linewidths=0,
                        xticklabels=xtick,
                        yticklabels=ytick)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, fontsize=12)
        ax.invert_yaxis()
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_title(title, fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)
        # plt.show()
        show_push_save(fig, func, args)

def plot_logistic_regression(data, args):
    """
    Plots accuracy score for different values of lambda and eta
    """
    for name, accuracy in data.items():
        accuracy = accuracy[:,:,-1]
        func = defaultdict(lambda: None)
        func["train"] = name.split()[0]
        fig, ax = plt.subplots()

        cols = np.around(np.log10(args.lmb), decimals = 3)
        idx = np.round(args.eta, 5)
        data = pd.DataFrame(accuracy, index=idx, columns=cols)
        xlabel = r"Hyper-parameter $\log(\lambda)$"
        ylabel = r"Learning rate $\eta$"
        title = name + f" for Cancer data, using {args.batch_size} minibatches"
        xtick = cols
        ytick = idx
        xrot = 0

        func["y"] = "eta"
        func["x"] = "lambda"
        func["z"] = "accuracy"
        ax = sns.heatmap(data,
                         ax=ax,
                         annot=True,
                         cmap=cm.coolwarm,
                         linewidths=0.1,
                         xticklabels=xtick,
                         yticklabels=ytick,
                         fmt=".3")

        from matplotlib.ticker import FormatStrFormatter
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, fontsize=12)
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.invert_yaxis()
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_title(title, fontsize=18)
        cbar = ax.collections[0].colorbar
        fig.set_size_inches(9, 8)
        cbar.ax.tick_params(labelsize=13)
        show_push_save(fig, func, args)


def plot_logistic_regression_epochs(data, args):
    """
    Plots accuracy score as a function of epochs and learnign rate
    """
    for name, accuracy in data.items():

        accuracy = accuracy[:, 0, :]
        func = defaultdict(lambda: None)
        func["train"] = name.split()[0]
        fig, ax = plt.subplots()
        
        cols = np.arange(args.num_epochs)
        idx = np.round(args.eta, 5)
        data = pd.DataFrame(accuracy, index=idx, columns=cols)
        xlabel = "Number of epochs"
        ylabel = r"Learning rate $\eta$"
        title = name + f" for Cancer data, using {args.batch_size} minibatches"
        xtick = len(cols)//10
        ytick = idx
        xrot = 0

        func["x"] = "epochs"
        func["y"] = "eta"
        func["z"] = "accuracy"
        ax = sns.heatmap(data,
                         ax=ax,
                         annot=False,
                         cmap=cm.coolwarm,
                         linewidths=0,
                         xticklabels=xtick,
                         yticklabels=ytick)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, fontsize=12)
        ax.invert_yaxis()
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_title(title, fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)
        show_push_save(fig, func, args)

def eta_lambda(data, args, NN=False, vmax=None):
    """
    Plots MSE or accuracy as heatmap for different etas and lambdas
    """
    for name, accuracy in data.items():
        print(name)
        func = defaultdict(lambda:None)
        func["train"] = name.split()[0]
        if args.dynamic_eta:
            func["x"] = "dynamic_eta"
        else:
            func["x"] = "eta"
        func["y"] = "lambda"
        func["z"] = name.split()[1]

        fig, ax = plt.subplots(figsize=(10, 10))
        try:
            col = np.round(np.log10(args.lmb), 3)
        except:
            col = np.round(args.lmb, 3)
        try:
            row = np.round(np.log10(args.eta), 3)
        except:
            row = np.round(args.eta, 3)
        data = pd.DataFrame(accuracy, index=row, columns=col)
        ax = sns.heatmap(data, 
                        ax=ax, 
                        vmax=vmax,
                        annot=True, 
                        cmap=cm.coolwarm)
        if NN:
            title = name + f" gridsearch, using {args.act_func.replace('_', ' ')}" + "\n" + f" activation function on {args.dataset}-data"
        else:
            title = name + " as function of " + r"$\eta$ and $\lambda$" + f" for {args.dataset}-data"
        
        ax.set_title(title, fontsize=18)
        
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
        ax.invert_yaxis()
        if args.dynamic_eta:
            ax.set_ylabel(r"$\log_{10}(\eta_0)$", fontsize=15)
        else:
            ax.set_ylabel(r"$\log_{10}(\eta)$", fontsize=15)

        ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=15)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)

        show_push_save(fig, func, args)

def eta_epochs(data, args, vmax=None, vmin=None):
    """
    Plots MSE or accuracy as function of epochs, for different etas, as a heatmap
    """
    for name, accuracy in data.items():
        func = defaultdict(lambda: None)
        func["train"] = name.split()[0]
        fig, ax = plt.subplots()

        cols = np.arange(args.num_epochs)
        idx = np.round(np.log10(args.eta),5)
        data = pd.DataFrame(accuracy, index=idx, columns=cols)

        xtick = len(cols)//10
        ytick = idx
        xrot = 0
        
        title = name + r" for Cancer data, using NN with 20 minibatches"

        if args.act_func == "relu":
            title += "\n" + r"Using the Relu activation function"
        elif args.act_func == "leaky_relu":
            title += "\n" + r"Using the Leaky Relu activation function"
        
        title = "NN test accuracy for Cancer data, with dynamic learning rate "
        xlabel = "Number of epochs"
        
        if args.dynamic_eta:
            title += "\n" + r"with dynamic learning rate, $\eta$"
            ylabel = r"Initial learning rate, $\eta_0$"
            func["y"] = "dynamic_eta"
        else:
            func["y"] = "eta"
            ylabel = r"Learning rate, $\eta$"
            
        func["x"] = "epochs"
        func["z"] = name.split()[1]

        ax = sns.heatmap(data,
                        ax=ax,
                        annot=False,
                        cmap=cm.coolwarm,
                        vmax=vmax,
                        vmin=vmin,
                        linewidths=0,
                        xticklabels=xtick,
                        yticklabels=ytick)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, fontsize=12)
        ax.invert_yaxis()
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_title(title, fontsize=18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)
        # plt.show()
        show_push_save(fig, func, args)


def train_history(NN, args):
    """
    Plots mse or accuracy as function of epochs, calculated during training
    """
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
