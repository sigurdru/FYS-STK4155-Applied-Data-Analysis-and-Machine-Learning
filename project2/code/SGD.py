import numpy as np
import utils 



def SGD(X, z, args, beta, eta, lmb=0):
    """
    Performs OLS regression using SGD

    Args:
        X, tuple: design matrix (train, test)
        z1, tuple: datapoints (train, test)
        args, argparse
        beta, 1darray: parameters
        eta, float: learning rate
        gamma, float: Momentum parameter
        lmb, float: For Ridge regression
    Returns:
        total_gradient 1darray: total gradient
    """

    n = int(X[0].shape[0])  # number of datapoints for training
    if args.batch_size == 0:
        M = n
    else:
        M = args.batch_size  # size of minibatch

    m = n // M
    v = 0
    X_train, X_test = X
    z_train, z_test = z

    inds = np.arange(0, n)

    MSE_train = np.zeros(args.num_epochs)
    MSE_test  = np.zeros(args.num_epochs)
    eta_0 = eta

    for epoch_i in range(args.num_epochs):
        # Initialize randomized training data for epoch
        np.random.shuffle(inds)
        X_train_shuffle = X_train[inds]
        z_train_shuffle = z_train[inds]

        for i in range(0, n, M):
            # Loop over mini batches

            xi = X_train_shuffle[i:i+M]
            zi = z_train_shuffle[i:i+M]

            gradient = 2 * xi.T @ ((xi @ beta)-zi.T[0]) / M \
                        + 2 * lmb * beta

            v = v * args.gamma + eta * gradient
            beta = beta - v

        train_pred = (X_train @ beta)
        test_pred = (X_test @ beta)

        MSE_train[epoch_i] = utils.MSE(z_train.T[0], train_pred)
        MSE_test[epoch_i] = utils.MSE(z_test.T[0], test_pred)

        # eta = learning_schedule(epoch_i*m + i,args)

    return MSE_train, MSE_test


def RidgeSG():
    return None


def learning_schedule(t, args):
    # if args.learning_schedule == True:
    #     t0 = 1
    #     t1 = 10
    #     # return args.t0/(t+args.t1)
    #     return t0/(t+t1)
    # else:
    #     return args.eta
    t0 = 10
    t1 = 1
    return t0/(t+t1)

