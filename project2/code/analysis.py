from collections import defaultdict
import numpy as np
# import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
# Our files
# from regression impo/rt Ordinary_least_squaresSG, RidgeSG
import utils
# import plot
from sklearn.linear_model import SGDRegressor
from NeuralNetwork import FFNN
import SGD
from collections import defaultdict
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


class NoneScaler(StandardScaler):
    """
    To have option of no scaling
    """

    def transform(self, x):
        return x


scale_conv = {"None": NoneScaler(), "S": StandardScaler(
    with_std=False), "N": Normalizer(), "M": MinMaxScaler()}


def NN_regression(args):
    p = args.polynomial
    etas = args.eta
    lmbs = args.lmb
    scaler = scale_conv[args.scaling]
    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p, intercept=False if p == 1 else True)
    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)

    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs)), dtype=float))
    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            NN = FFNN(X_train,
                      z_train,
                      hidden_nodes=args.hidden_nodes,
                      batch_size=args.batch_size,
                      learning_rate=eta,
                      lmb=lmb,
                      )
            NN.train(args.num_epochs)

            train_pred = NN.predict(X_train)
            test_pred = NN.predict(X_test)

            tr_mse = utils.MSE(z_train, train_pred)
            te_mse = utils.MSE(z_test, test_pred)

            data["train_MSE"][i][j] = tr_mse if tr_mse < 1 else np.nan
            data["test_MSE"][i][j] = te_mse if te_mse < 1 else np.nan

    print("\n"*3)
    print(f"Best NN train prediction: {(train:=data['train_MSE'])[(mn:=np.unravel_index(np.nanargmin(train), train.shape))]} for eta = {np.log10(etas[mn[0]])}, lambda = {lmbs[mn[1]]}")
    print(f"Best NN test prediction: {(test:=data['test_MSE'])[(mn:=np.unravel_index(np.nanargmin(test), test.shape))]} for eta = {np.log10(etas[mn[0]])}, lambda = {lmbs[mn[1]]}")

    for name, accuracy in data.items():
        fig, ax = plt.subplots()
        data = pd.DataFrame(accuracy, index=np.log10(etas), columns=np.log10(lmbs))
        sns.heatmap(data, ax=ax, annot=True, cmap=cm.coolwarm)
        ax.set_title(name)
        ax.set_ylabel("$\log10(\eta)$")
        ax.set_xlabel("$\log10(\lambda)$")
        plt.show()


def linear_regression(args):
    p = args.polynomial
    etas = args.eta
    lmbs = args.lmb

    if args.batch_size == 0:
        batch_sizes = np.array([720, 360, 240, 144, 72, 48, 30, 24])
    else:
        batch_sizes = np.array([args.batch_size])

    scaler = scale_conv[args.scaling]
    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p)

    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)

    n_minibatches = X_train.shape[0] // batch_sizes
    ols_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    ols_pred = X_test @ ols_beta


    data = defaultdict(lambda: np.zeros((len(etas),
                                        len(lmbs),
                                        len(batch_sizes),
                                        args.num_epochs), dtype=float))

    beta0 = np.random.randn(utils.get_features(p))

    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            for k, batch_size in enumerate(batch_sizes):
                MSE_train, MSE_test = SGD.SGD((X_train, X_test),
                                                (z_train, z_test),
                                                args,
                                                beta0,
                                                eta,
                                                batch_size,
                                                lmb)

                data["Train MSE"][i][j][k] = MSE_train
                data["Test MSE"][i][j][k] = MSE_test

    if not args.gamma:
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
            if len(lmbs) == 1 and len(batch_sizes) == 1:
                """
                (x,y): (epochs, eta values)

                Good simulation:
                - Ne  : 150
                - eta : np.linspace(0.01, 0.05, 31)
                """
                cols = np.arange(args.num_epochs)
                idx = np.round(etas, 3)
                data = pd.DataFrame(accuracy[:,0,0,:], index=idx, columns=cols[:])
                ylabel = "Learning rate $\eta$"
                xlabel = "Number of epochs"
                title = name + " for Franke function, using 20 minibatches"
                ytick = 3
                vmax = 0.07

            if len(lmbs) == 1 and len(etas) == 1:
                """
                (x,y): (epochs, number of minibatches)

                Good simulation:
                - Ne : 150
                - eta: 0.3
                - bs : 0
                """
                cols = np.arange(args.num_epochs)
                idx = n_minibatches
                data = pd.DataFrame(accuracy[0,0,:,:], index=idx, columns=cols[:])
                ylabel = "Number of minibatches"
                xlabel = "Number of epochs"
                title = name + " for Franke function, using $\eta=0.3$"
                ytick = idx
                vmax = 0.07

            if len(lmbs) != 1 and len(n_minibatches) == 1:
                """
                (x,y): (log10[lambda], eta values)

                Good simulation:
                - Ne  : 200
                - lmb : np.logspace(-5, 0, 11)
                - eta : np.linspace(0.1, 0.7, 11)
                """
                cols = np.log10(lmbs)
                idx = np.round(etas, 3)
                data = pd.DataFrame(accuracy[:,:,0,-1], index=idx, columns=cols[:])
                ylabel = "Learning rate $\eta$"
                xlabel = "$\log_{10}(\lambda)$"
                title = name + " for Franke function after 200 epochs, using 20 minibatches"
                ytick = idx
                vmax = None


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
            plt.show()

    else:
        gammas = np.array([0.1, 0.25, 0.5, 0.9])

        for gamma in gammas:
            plt.plot(data["Train MSE"].flatten(), label='Train MSE. $\gamma=0$')
            plt.plot(data["Test MSE"].flatten() , label='Test MSE. $\gamma=0$')

            mom_MSE_train, mom_MSE_test = SGD.SGD((X_train, X_test),
                                                (z_train, z_test),
                                                args,
                                                beta0,
                                                eta,
                                                batch_size,
                                                lmb,
                                                gamma=gamma)
            plt.plot(mom_MSE_train, '--', label='Train MSE. $\gamma={:.2f}$'.format(gamma))
            plt.plot(mom_MSE_test , '--', label='Test MSE. $\gamma={:.2f}$'.format(gamma))
            plt.legend()
            plt.show()



def logistic_regression(args):
    pass


def NN_classification(args):
    dataset = utils.load_data(args)
    X, z_ = dataset.data, dataset.target.reshape(-1, 1)
    z = utils.categorical(z_)
    scaler = scale_conv[args.scaling]
    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)
    print(X_train)

    NN = FFNN(X_train,
              z_train,
              hidden_nodes=args.hidden_nodes,
              batch_size=args.batch_size,
              learning_rate=args.eta[0],
              lmb=args.lmb[0],
              clas=True,
              activation="leaky_relu",
              cost="accuracy",
              )
    NN.train(args.num_epochs)

    probs = NN.predict(X_train)
    pred = np.argmax(probs, axis=1).reshape(-1, 1)
    print(np.sum(pred == z_train) / len(z_train))

    probs = NN.predict(X_test)
    pred = np.argmax(probs, axis=1).reshape(-1, 1)
    print(np.sum(pred == z_test) / len(z_test))

