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
    lmbs = args.lmb # Default 0

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

                data["train_MSE"][i][j][k] = MSE_train
                data["test_MSE"][i][j][k] = MSE_test

                # plt.plot(data['train_MSE'][i][j], label='train')
                # plt.plot(data['test_MSE'][i][j], label='test')
                # plt.legend()
                # plt.show()

    for name, accuracy in data.items():
        fig, ax = plt.subplots()
        cols = np.arange(args.num_epochs)
        if len(lmbs) == 1 and len(batch_sizes) == 1:
            data = pd.DataFrame(accuracy[:,0,0,:], index=np.round(etas, 3), columns=cols[:])
            ax.set_ylabel("$\eta$")
            ax.set_xlabel("Number of epochs")


        if len(lmbs) == 1 and len(etas) == 1:
            # n_mbs =
            data = pd.DataFrame(accuracy[0,0,:,:], index=n_minibatches, columns=cols[:])
            ax.set_ylabel("Number of minibatches")
            ax.set_xlabel("Number of epochs")

        ax = sns.heatmap(data,
                        ax=ax,
                        annot=False,
                        cmap=cm.coolwarm,
                        vmax=0.07,
                        linewidths=0)
        ax.invert_yaxis()
        ax.set_title(name)
        plt.show()


def logistic_regression(args):
    pass


def NN_classification(args):
    if args.dataset == "Cancer":
        cancer = utils.load_data(args)
        X, z = cancer.data, cancer.target.reshape(-1, 1)
        # z = z.reshape(-1, 1)
        print(X.shape)
        print(z.shape)
        for i, eta in enumerate(args.eta):
            for j, lmb in enumerate(args.lmb):
                NN = FFNN(X,
                          z,
                          hidden_nodes=args.hidden_nodes,
                          batch_size=args.batch_size,
                          learning_rate=eta,
                          lmb=lmb,
                          clas=True,
                          activation="sigmoid",
                          cost="accuracy",
                          )
                NN.train(args.num_epochs)
                print(NN.weights)

    else:
        train, test = utils.load_data(args)
        # X_train, z_train = train.
