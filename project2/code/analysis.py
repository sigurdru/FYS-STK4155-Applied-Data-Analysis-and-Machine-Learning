import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split as tts
#Our files
from regression import Ordinary_least_squaresSG, RidgeSG
import utils
import plot
from sklearn.linear_model import SGDRegressor

class NoneScaler(StandardScaler):
    """ 
    To have option of no scaling
    """
    def transform(self, x):
        return x
# Dicts converting from string to callable functions
reg_conv = {"OLS": Ordinary_least_squaresSG, "Ridge": RidgeSG}
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
    p = args.polynomial                     #polynomial degree
    n_epochs = args.num_epochs              #number of epochs
    M = args.minibatch                      #size of minibatch
    n = int((args.num_points**2)*(1-tts))   #number of datapoints for testing
    m = int(n/M)                            # number of minibatches

    scaler = scale_conv[args.scaling]
    reg_method = reg_conv[args.method]
    x, y, z = utils.load_data(args)

    #create design matrix
    X = utils.create_X(x, y, p)
    #Split and scale data
    X_train, X_test, z_train, z_test = split_scale(X, z, args.tts, scaler)
    test_MSE = []
    train_MSE = []
    inds = np.arange(0, n)

    for eta in args.eta:
        # beta = (np.linalg.pinv(X.T @ X) @ X.T @ z)[:,0]

        beta = np.random.randn(utils.get_features(p))
        # print('MSE = %f with eta = %f' %(MSE(z_test.T[0], pred), eta))

        for epoch_i in range(n_epochs):
            # Initialize randomized training data for epoch 
            np.random.shuffle(inds)
            X_train_shuffle = X_train[inds]
            z_train_shuffle = z_train[inds]

            for i in range(0, n, M):
                # Loop over mini batches 

                # random_index = np.random.randint(n-M)
                # xi = X_train[random_index:random_index + M]
                # zi = z_train[random_index:random_index + M]

                xi = X_train_shuffle[i:i+M]
                zi = z_train_shuffle[i:i+M]

                gradient = reg_method(xi, zi, args, beta, eta, epoch_i, i)
                beta = beta - eta * gradient
            
            train_pred = (X_train @ beta)
            test_pred = (X_test @ beta)
            
            train_MSE.append(MSE(z_train.T[0], train_pred))
            test_MSE.append(MSE(z_test.T[0], test_pred))
            # print('MSE = %f with eta = %f' %(MSE(z_test.T[0], pred), eta))

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

    import matplotlib.pyplot as plt
    plt.plot(train_MSE, label='train')
    plt.plot(test_MSE, label='test')
    plt.legend()
    plt.show()

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

def MSE(y, y_pred):
    return sum((y - y_pred) ** 2) / len(y)
