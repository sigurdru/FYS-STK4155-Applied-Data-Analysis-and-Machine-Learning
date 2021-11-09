from collections import defaultdict
import enum
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
import plot


class NoneScaler(StandardScaler):
    """
    To have option of no scaling
    """

    def transform(self, x):
        return x


scale_conv = {"None": NoneScaler(),
                "S": StandardScaler(),
                "N": Normalizer(),
                "M": MinMaxScaler()}


def NN_regression(args):
    p = args.polynomial
    etas = args.eta
    lmbs = args.lmb
    scaler = scale_conv[args.scaling]
    if args.pred:
        # Reduce noise for surface plot comparison
        args.epsilon = 0.05
    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p, intercept=False if p == 1 else True)
    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)

    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs)), dtype=float))
    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            np.random.seed(args.seed)
            NN = FFNN(X_train,
                      z_train,
                      hidden_nodes=args.hidden_nodes,
                      batch_size=args.batch_size,
                      learning_rate=eta,
                      dynamic_eta=args.dynamic_eta,
                      lmb=lmb,
                      )
            NN.train(args.num_epochs, train_history=args.pred, test=(X_test, z_test))

            # Rescale data to obtain correct values
            train_pred = utils.rescale_data(NN.predict(X_train), z)
            test_pred = utils.rescale_data(NN.predict(X_test), z)

            z_train_ = utils.rescale_data(z_train, z)
            z_test_ = utils.rescale_data(z_test, z)

            tr_mse = utils.MSE(z_train_, train_pred)
            te_mse = utils.MSE(z_test_, test_pred)

            print('mse train  : ', tr_mse)
            print('mse test   : ', te_mse)

            # plt.plot(range(args.num_epochs), hist)
            # plt.show()

            data["train MSE"][i][j] = tr_mse if tr_mse < 1 else np.nan
            data["test MSE"][i][j] = te_mse if te_mse < 1 else np.nan

            if args.pred:
                """
                Plot fitted surface with original data
                """
                X_ = utils.split_scale(X, z, 0, scaler)[0] # Scale design matrix

                # Calculate output. Rescale values to the original
                z_pred = utils.rescale_data(NN.predict(X_), z)
                
                # Plot result of fit and exit
                plot.surface_fit(z_pred, z, x, y, args)

    print("\n"*3)
    print(f"Best NN train prediction: {(train:=data['train MSE'])[(mn:=np.unravel_index(np.nanargmin(train), train.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
    print(f"Best NN test prediction: {(test:=data['test MSE'])[(mn:=np.unravel_index(np.nanargmin(test), test.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
    plot.eta_lambda(data, args)


def linear_regression(args):
    print("Doing linear regression")
    p = args.polynomial
    etas = args.eta
    lmbs = args.lmb

    if args.pred:
        # Reduce noise for surface plot comparison
        args.epsilon = 0.05

    if args.scaling == "S":
        # Don't divide data by std for Franke 
        # Since std<1, causing too much increase of data values  
        scaler = StandardScaler(with_std=False)
    else:
        scaler = scale_conv[args.scaling]

    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p, intercept=False if p == 1 else True)

    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)
    beta0 = np.random.randn(utils.get_features(p))  # use same beta0 in all runs for comparisonability
    if args.gamma >= 0:
        if args.batch_size == 0:
            batch_sizes = np.array([720, 360, 240, 144, 72, 48, 36, 30, 24])
        else:
            batch_sizes = np.array([args.batch_size])
        args.batch_size = batch_sizes

        n_minibatches = X_train.shape[0] // batch_sizes
        args.nmb = n_minibatches

        data = defaultdict(lambda: np.zeros((len(etas),
                                            len(lmbs),
                                            len(batch_sizes),
                                            args.num_epochs), dtype=float))


        for i, eta in enumerate(etas):
            for j, lmb in enumerate(lmbs):
                for k, batch_size in enumerate(batch_sizes):
                    # np.random.seed(args.seed)
                    MSE_train, MSE_test, beta = SGD.SGD((X_train, X_test),
                                                    (z_train, z_test),
                                                    args,
                                                    beta0,
                                                    eta,
                                                    batch_size,
                                                    lmb,
                                                    args.gamma)

                    data["Train MSE"][i][j][k] = MSE_train
                    data["Test MSE"][i][j][k] = MSE_test

                    if args.pred:
                        """
                        Plot fitted prediction vs data set with epsilon=0.05
                        """
                        # Scale the full design matrix 
                        X_ = utils.split_scale(X, z, 0, scaler)[0] # Scale design matrix

                        # Calculate prediction. Rescale values 
                        z_pred = X_ @ beta + np.mean(z)

                        # Plot result of fit, and exit
                        plot.surface_fit(z_pred, z, x, y, args)

        plot.parameter_based(data, args)

    else:
        gammas = np.linspace(0, 0.95, 20)
        args.gamma = gammas
        data = defaultdict(lambda: np.zeros((len(gammas), args.num_epochs)))

        for i, gamma in enumerate(gammas):

            mom_MSE_train, mom_MSE_test, beta = SGD.SGD((X_train, X_test),
                                                (z_train, z_test),
                                                args,
                                                beta0,
                                                args.eta[0],
                                                args.batch_size,
                                                args.lmb[0],
                                                gamma=gamma)

            data["Train MSE"][i] = mom_MSE_train
            data["Test MSE"][i] = mom_MSE_test

        plot.momentum(data, args)


def logistic_regression(args):
    pass


def NN_classification(args):
    etas = args.eta
    lmbs = args.lmb

    dataset = utils.load_data(args)
    z_ = dataset.target.reshape(-1, 1)
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    corrmat = data.corr()
    if args.dataset == "Cancer":
        feat = "mean concavity"
        X = data.loc[:, lambda x: abs(corrmat[feat]) < 0.8]
        X.insert(0, feat, data[feat])
    else:
        X = dataset.data
    print(X.shape)
    
    z = utils.categorical(z_)
    scaler = scale_conv[args.scaling]
    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)

    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs))))
    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            np.random.seed(args.seed)

            NN = FFNN(X_train,
                      z_train,
                      hidden_nodes=args.hidden_nodes,
                      batch_size=args.batch_size,
                      learning_rate=eta,
                      dynamic_eta=args.dynamic_eta,
                      lmb=lmb,
                      gamma=args.gamma,
                      wi=args.weight_initialization,
                      activation=args.act_func,
                      cost="cross_entropy",
                      output_activation="softmax",
                      )
            NN.train(args.num_epochs, train_history=args.pred, test=(X_test, z_test))
            data["train accuracy"][i][j] = NN.predict_accuracy(X_train, z_train)
            data["test accuracy"][i][j] = NN.predict_accuracy(X_test, z_test)
            print("Test accuracy: ", data["test accuracy"][i][j])
            print()

            if args.pred:
                plot.train_history(NN, args)
                exit()

    print(f"Best NN train prediction: {(train:=data['train accuracy'])[(mn:=np.unravel_index(np.nanargmax(train), train.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
    print(f"Best NN test prediction: {(test:=data['test accuracy'])[(mn:=np.unravel_index(np.nanargmax(test), test.shape))]} for eta = {etas[mn[0]]}, lambda = {lmbs[mn[1]]}")
    plot.eta_lambda(data, args, NN=True)
