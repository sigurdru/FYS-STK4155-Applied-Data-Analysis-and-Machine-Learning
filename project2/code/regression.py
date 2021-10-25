import numpy as np

def Ordinary_least_squaresSG(xi, zi, args, beta, eta, epoch_i, i,lmb=0):
    """
    Performs OLS regression using SGD

    Args:
        x1, 2darray: design matrix
        z1, 1darray: datapoints
        beta, 1darray: parameters
        epoch_i, int: epoch iteration
        eta, float: learning rate
        args, argparse
        lmb, any: taken for compatibility reasons. Unused
    Returns:
        total_gradient 1darray: total gradient
    """
    tts = args.tts  # train test split
    M = args.minibatch  # size of minibatch
    n = int(args.num_points*(1-tts))**2  # number of datapoints for testing
    m = int(n/M)  # number of minibatches
    gradients = 2.0 * xi.T @ ((xi @ beta)-zi)
    # eta = learning_schedule(epoch_i*m + i,args)
    total_gradient = np.sum(gradients, axis=1)
    return total_gradient


def RidgeSG(xi, zi, args, beta, eta, epoch_i, i, lmb=0):
    """
    Performs Ridge regression using SGD

    Args:
        x1, 2darray: design matrix
        z1, 1darray: datapoints
        beta, 1darray: parameters
        epoch_i, int: epoch iteration
        eta, float: learning rate
        lmb, float: hyper-parameter 
        args, argparse
    Returns:
        total_gradient 1darray: total gradient
    """
    tts = args.tts  # train test split
    M = args.minibatch  # size of minibatch
    n = int(args.num_points*(1-tts))**2  # number of datapoints for testing
    m = int(n/M)  # number of minibatches
    gradients = 2.0 * xi.T @ ((xi @ beta)-zi) + 2*lmb*beta
    # eta = learning_schedule(epoch_i*m + i,args)
    total_gradient = eta * np.sum(gradients, axis=1)
    return total_gradient


# def learning_schedule(t, args):
#     if args.learning_schedule == True:
#         t0 = 1
#         t1 = 10
#         # return args.t0/(t+args.t1)
#         return t0/(t+t1)
#     else:
#         return args.eta
