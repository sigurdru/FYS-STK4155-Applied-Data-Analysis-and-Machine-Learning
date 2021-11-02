from collections import defaultdict
import numpy as np
# import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split as tts
# Our files
# from regression impo/rt Ordinary_least_squaresSG, RidgeSG
import utils
# import plot
from sklearn.linear_model import SGDRegressor
from NeuralNetwork import FFNN
from collections import defaultdict
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from SGD import SGD

class NoneScaler(StandardScaler):
    """ 
    To have option of no scaling
    """

    def transform(self, x):
        return x


scale_conv = {"None": NoneScaler(), "S": StandardScaler(
    with_std=False), "N": Normalizer(), "M": MinMaxScaler()}


def split_scale(X, z, ttsplit, scaler):
    """
    Split and scale data
    Also used to scale data for CV, but this does its own splitting.
    Args:
        X, 2darray: Full design matrix
        z, 2darray: dataset
        ttsplit, float: train/test split ratio
        scaler, sklearn.preprocessing object: Is fitted to train data, scales train and test
    Returns:
        X_train, X_test, z_train, z_test, 2darrays: Scaled train and test data
    """

    if ttsplit != 0:
        X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)
    else:
        X_train = X
        z_train = z
        X_test = X
        z_test = z

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler.fit(z_train)
    z_train = scaler.transform(z_train)
    z_test = scaler.transform(z_test)

    return X_train, X_test, z_train, z_test


def analyse_NN(args):
    p = args.polynomial
    etas = args.eta
    # lmbs = np.ones(5)
    lmbs = [0,]
    scaler = scale_conv[args.scaling]
    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p)
    X_train, X_test, z_train, z_test = split_scale(X, z, args.tts, scaler)
    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs)), dtype=float))

    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            NN = FFNN(X_train,
                      z_train,
                      hidden_nodes=[10,10],
                      batch_size=args.batch_size,
                      learning_rate=eta,
                      lmb=lmb,
                      )
            NN.train(args.num_epochs) # Default 100 

            train_pred = NN.predict(X_train)
            test_pred = NN.predict(X_test)

            print('eta={:.2f}: MSE_test={:.3f}'.format(eta, MSE(z_test,test_pred)[0]))
            data["train_accuracy"][i][j] = MSE(z_train, train_pred)
            data["test_accuracy"][i][j] = MSE(z_test, test_pred)


    for name, accuracy in data.items():
        plt.plot(etas, data[name], 'o-', label=name)
        fig, ax = plt.subplots()
        sns.heatmap(accuracy, ax=ax, annot=True)
        # ax.set_title(name)
        ax.set_xlabel("$\eta$")
        # ax.setylabel("$\lambda$")
        # plt.show()
    plt.legend()
    plt.show()


def analyse_SGD(args):
    """
    To be completed 
    """
    p = args.polynomial
    etas = args.eta
    lmbs = [0,]
    scaler = scale_conv[args.scaling]
    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p)
    X_train, X_test, z_train, z_test = split_scale(X, z, args.tts, scaler)
    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs)), dtype=float))
    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            NN = FFNN(X_train,
                      z_train,
                      hidden_nodes=[10,10],
                      batch_size=args.batch_size,
                      learning_rate=eta,
                      lmb=lmb,
                      )

            # rnd_seed = np.random.randint(0,1000)
            np.random.seed(69420)
            NN.train(args.num_epochs)
            print('finished training')
            
            # train_pred = NN.predict(X_train)
            # test_pred = NN.predict(X_test)
            X_ = split_scale(X, z, 0, scaler)[0]
            z_ = NN.predict(X_) 
            z_NN = z_.reshape((x.shape[0], x.shape[1]))
            z_MSE = MSE(z, z_)


            # OLS_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
            # OLS_train_pred = X_train @ OLS_beta 
            # OLS_test_pred = X_test @ OLS_beta
            # z_OLS = (X @ OLS_beta).reshape((x.shape[0], x.shape[1]))

            np.random.seed(100)
            beta0 = np.random.randn(utils.get_features(p)) 
            # beta_SGD = SGD((X_train, X_test), (z_train, z_test), args, beta0, eta, gamma=0)
            # tr_m,te_m = SGD((X_train, X_test), (z_train, z_test), args, beta0, eta, gamma=args.gamma)

            # z_SGD = (X_ @ beta_SGD).reshape((x.shape[0], x.shape[1]))

            fig = plt.figure()
            ax = fig.gca(projection="3d")
            # Plot the surface.
            # surf = ax.plot_surface(x, y, z_SGD, cmap=cm.coolwarm,linewidth=0, antialiased=False);ax.set_title('SGD')
            surf = ax.plot_surface(x, y, z_NN, cmap=cm.coolwarm, linewidth=0, antialiased=False);ax.set_title('NN')
            # surf = ax.plot_surface(x, y, z_OLS, cmap=cm.coolwarm, linewidth=0, antialiased=False);ax.set_title('OLS')

            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()
            exit()

def MSE(z, ztilde):
    return sum((z - ztilde)**2) / len(z)