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
import pandas as pd

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


def analyse(args):
    p = args.polynomial
    etas = args.eta
    # lmbs = np.ones(5)
    lmbs = [0,]
    scaler = scale_conv[args.scaling]
    x, y, z = utils.load_data(args)
    X = utils.create_X(x, y, p)
    X_train, X_test, z_train, z_test = split_scale(X, z, args.tts, scaler)

    ols_beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    ols_pred = X_test @ ols_beta

    print(ols_pred.shape)

    data = defaultdict(lambda: np.zeros((len(etas), len(lmbs)), dtype=float))
    for i, eta in enumerate(etas):
        for j, lmb in enumerate(lmbs):
            NN = FFNN(X_train,
                      z_train,
                      hidden_nodes=[10,10],
                      batch_size=args.num_points,
                      learning_rate=eta,
                      lmb=lmb,
                      verbose=True,
                      )
            NN.train(1000)
            # print('finished training')

            train_pred = NN.predict(X_train)
            test_pred = NN.predict(X_test)


            # print('eta={:.2f}: MSE_test={:.3f}'.format(eta, MSE(z_test,test_pred)[0]))
            data["train_MSE"][i][j] = MSE(z_train, train_pred)
            data["test_MSE"][i][j] = MSE(z_test, test_pred)
            
    print("\n"*3)
    print(f"Best NN train prediction: {(train:=data['train_MSE'])[(mn:=np.unravel_index(np.argmin(train), train.shape))]} for eta = {np.log10(etas[mn[0]])}, lambda = {lmbs[mn[1]]}")
    print(f"Best NN test prediction: {(test:=data['test_MSE'])[(mn:=np.unravel_index(np.argmin(test), test.shape))]} for eta = {np.log10(etas[mn[0]])}, lambda = {lmbs[mn[1]]}")
    print(f"OLS test prediction: {MSE(z_test, ols_pred)}")
    for name, accuracy in data.items():
        fig, ax = plt.subplots()
        data = pd.DataFrame(accuracy, index=np.log10(etas), columns=lmbs)
        sns.heatmap(data, ax=ax, annot=True)
        ax.set_title(name)
        ax.set_ylabel("$\log10(\eta)$")
        ax.set_xlabel("$\lambda$")
        plt.show()

def MSE(z, ztilde):
    return sum((z - ztilde)**2) / len(z)
# def SGD(args):
#     for eta in self.args.eta:
#         beta = np.random.randn(utils.get_features(self.p))

#         for epoch_i in range(self.args.n_epochs):
#             pass
