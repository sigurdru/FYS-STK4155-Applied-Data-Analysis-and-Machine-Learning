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
            hist, err = NN.train(args.num_epochs, train_history=False)

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
                Should be inputs to a separate plot script in the future
                """
                X_ = utils.split_scale(X, z, 0, scaler)[0] # Scale design matrix 
                z_ = utils.split_scale(X, z, 0, scaler)[-1]# Scale targets 

                # Calculate output. Rescale values to the original 
                z_pred = utils.rescale_data(NN.predict(X_), z)
                z_pred_ = z_pred.reshape((x.shape[0], x.shape[1]))

                z_target = z.reshape((x.shape[0], x.shape[1]))

                fig = plt.figure()
                ax = fig.gca(projection="3d")
                surf = ax.plot_surface(x,y,z_target, alpha=0.3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                surf = ax.plot_surface(x,y,z_pred_, cmap=cm.jet, linewidth=0, antialiased=False)

                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.show()

    # print("\n"*3)
    # print(f"Best NN train prediction: {(train:=data['train_MSE'])[(mn:=np.unravel_index(np.nanargmin(train), train.shape))]} for eta = {np.log10(etas[mn[0]])}, lambda = {lmbs[mn[1]]}")
    # print(f"Best NN test prediction: {(test:=data['test_MSE'])[(mn:=np.unravel_index(np.nanargmin(test), test.shape))]} for eta = {np.log10(etas[mn[0]])}, lambda = {lmbs[mn[1]]}")
    plot.eta_lambda(data, args)


def linear_regression(args):
    print("Doing linear regression")
    p = args.polynomial
    etas = args.eta
    lmbs = args.lmb
    scaler = scale_conv[args.scaling]
    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p, intercept=False if p == 1 else True)

    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)
    beta0 = np.random.randn(utils.get_features(p))  # use same beta0 in all runs for comparisonability
    if args.gamma >= 0:
        if args.batch_size == 0:
            batch_sizes = np.array([720, 360, 240, 144, 72, 48, 30, 24])
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
                        Should be inputs to a separate plot script in the future
                        """
                        X_ = utils.split_scale(X, z, 0, scaler)[0] # Scale design matrix 


                        # Calculate output. Rescale values to the original 
                        z_pred = utils.rescale_data(X_ @ beta, z)
                        z_pred_ = z_pred.reshape((x.shape[0], x.shape[1]))

                        z_target = z.reshape((x.shape[0], x.shape[1]))

                        fig = plt.figure()
                        ax = fig.gca(projection="3d")
                        surf = ax.plot_surface(x,y,z_target, alpha=0.3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                        surf = ax.plot_surface(x,y,z_pred_, cmap=cm.jet, linewidth=0, antialiased=False)

                        fig.colorbar(surf, shrink=0.5, aspect=5)
                        plt.show()

        plot.parameter_based(data, args)
        
    else:
        gammas = np.array([0, 0.25, 0.5, 0.9])
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

def classify(x):
    return np.round(x * 2)



def NN_classification(args):
    dataset = utils.load_data(args)
    X, z_ = dataset.data, dataset.target.reshape(-1, 1)
    z = utils.categorical(z_)
    scaler = scale_conv[args.scaling]
    X_train, X_test, z_train, z_test = utils.split_scale(X, z, args.tts, scaler)

    NN = FFNN(X_train,
              z_train,
              hidden_nodes=args.hidden_nodes,
              batch_size=args.batch_size,
              learning_rate=args.eta[0],
              lmb=args.lmb[0],
              activation="sigmoid",
              cost="cross_entropy",
              output_activation="softmax"
              )
    # NN.train(args.num_epochs)
    # exit()
    prob_hist, prob_err = NN.train(args.num_epochs, train_history=True)
    train_output = NN.predict(X_train)
    train_pred = np.argmax(train_output, axis=1)
    train_target = np.argmax(z_train, axis=1)

    test_output = NN.predict(X_test)
    test_pred = np.argmax(test_output, axis=1)
    test_target = np.argmax(z_test, axis=1)
    print('Train acc: ', np.sum(train_pred == train_target)/len(train_pred))
    print('Test acc : ', np.sum(test_pred == test_target) / len(test_pred))
    print("\n"*2)

    # Plot of accuracy as a function of epochs 
    plt.plot(np.arange(args.num_epochs), prob_hist, 'o-')
    plt.show()

    # Plot of error in output as a function of epochs
    plt.plot(np.arange(args.num_epochs), prob_err)
    plt.show()

