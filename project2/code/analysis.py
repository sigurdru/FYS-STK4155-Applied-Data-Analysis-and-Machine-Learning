import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split as tts
#Our files
from regression import SGD, RidgeSG
import utils
import plot
from sklearn.linear_model import SGDRegressor
import sys
import matplotlib.pyplot as plt 


class NoneScaler(StandardScaler):
    """ 
    To have option of no scaling
    """
    def transform(self, x):
        return x

# Dicts converting from string to callable functions
reg_conv = {"OLS": SGD, "Ridge": RidgeSG}
scale_conv = {"None": NoneScaler(), "S": StandardScaler(with_std=False), "N": Normalizer(), "M": MinMaxScaler()}


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

def analyze_SGD(args):
    """
    Denne funksjonen er midlertidlig.
    Prøver bare å lage en vanlig SGD med batches som funker
    på FrankeFunction
    """
    #Setting up parameters and data
    tts = args.tts                          #train test split
    P = args.polynomial                     #polynomial degree

    scaler = scale_conv[args.scaling]
    reg_method = reg_conv[args.method]
    x, y, z = utils.load_data(args)

    #create design matrix
    X = utils.create_X(x, y, P)

    #Split and scale data
    X_train, X_test, z_train, z_test = split_scale(X, z, tts, scaler)

    # To be implemented
    # for i, p in enumerate(P):


    for eta in args.eta:

        beta = np.random.randn(utils.get_features(P))    
        inputs = ((X_train, X_test), (z_train, z_test), args, beta, eta)
        MSE_train, MSE_test = reg_method(*inputs)

        MSE_train_mom, MSE_test_mom = reg_method(*inputs, gamma=args.gamma) 

    # Plotting MSE as a function of epochs 
    plt.plot(MSE_train, label='train')
    plt.plot(MSE_test, label='test')
    plt.plot(MSE_train_mom, '--', label='train momentum')
    plt.plot(MSE_test_mom, '--', label='test momentum')

    plt.legend()
    plt.show()

    # Comparing our own beta-values and MSE with sklearn 
    sgdreg = SGDRegressor(max_iter=100, penalty=None, eta0=args.eta[0])
    sgdreg.fit(X, z)
    print('beta from own sgd: ', len(beta))
    print(beta, '\n')
    print('MSE own: ')
    print(MSE(z_test.T[0], X_test @ beta), '\n')

    print('beta from sklearn: ', len(sgdreg.coef_))
    print(sgdreg.coef_, '\n')
    print('MSE sklearn:')
    print(MSE(z_test.T[0], X_test @ sgdreg.coef_))


# def analyze_MSGD(args):
#     """
#     Denne funksjonen er midlertidlig.
#     Prøver bare å lage en vanlig SGD med batches som funker
#     på FrankeFunction
#     """
#     #Setting up parameters and data
#     tts = args.tts  # train test split
#     p = args.polynomial  # polynomial degree
#     n_epochs = args.num_epochs  # number of epochs
#     eta = args.eta  # learning rate
#     M = args.minibatch  # size of minibatch
#     n = int(args.num_points*(1-tts))**2  # number of datapoints for testing
#     m = int(n/M)  # number of minibatches

#     scaler = scale_conv[args.scaling]
#     x, y, z = utils.load_data(args)

#     #create design matrix
#     X = utils.create_X(x, y, p)
#     #Split and scale data
#     X_train, X_test, z_train, z_test = split_scale(X, z, args.tts, scaler)
#     beta = np.random.rand(utils.get_features(p))
#     indices = np.arange(n)
#     for epoch_i in range(n_epochs):
#         random_index = np.random.choice(indices, size=(M), replace=True)
#         xi = X_train[random_index]
#         zi = z_train[random_index]
#         gradients = 2.0 * xi.T @ ((xi @ beta)-zi)
#         eta = learning_schedule(epoch_i*m, args)
#         total_gradient = np.sum(eta*gradients, axis=1)
#         beta -= total_gradient
#     pred = (X_test @ beta)
#     print(MSE(z_test.T[0], pred))


