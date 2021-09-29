import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split as tts
import utils
from collections import defaultdict
import regression
import resampling
import plot

reg_conv = {"OLS": regression.Ordinary_least_squares, "Ridge": regression.Ridge, "Lasso":regression.Lasso}
resampling_conv = {"None": resampling.NoResampling, "Boot": resampling.Bootstrap, "CV": resampling.cross_validation}
scale_conv = {"S": StandardScaler(), "N": Normalizer(), "M": MinMaxScaler()}


def simple_regression(args):
    """
    Run regression. Basically Ex1
    For Ridge and Lasso, a single lambda-value is used

    Can be used with any regression method
    Can be used with any resampling method

    Plot MSE for train and test as function of complexity
    """
    N = args.num_points
    P = args.polynomial  # polynomial degrees
    scaler = scale_conv[args.scaling]

    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, eps0=args.epsilon)

    MSEs = np.zeros(len(P))
    MSE_train = np.zeros(len(P))
    R2s = np.zeros(len(P))
    R2_train = np.zeros(len(P))

    resampl = resampling_conv[args.resampling]

    for i, p in enumerate(P):
        print("p =", p)
        X = utils.create_X(x, y, p)

        data = resampl(X, z, args.tts, args.resampling_iter, args.lmb[0], reg_conv[args.method], scaler)
        MSEs[i] = data["test_MSE"]
        MSE_train[i] = data["train_MSE"]
        # R2s[i] = data["test_R2"]
        # R2_train[i] = data["train_R2"]

    # Plotting the error, see output folder!
    plot.Plot_error(MSE_test=MSEs, MSE_train=MSE_train, args=args)
    # plot.Plot_R2(R2_test=R2s, R2_train=R2_train, args=args)


def bias_var_tradeoff(args):
    """
    Perform bias-variance trade-off analysis
    For Ridge and Lasso, a single lambda-value is used

    Can be used with any regression method
    Should be used with Bootstrapping as resampling method

    Plots MSE, bias and variance for train and test as function of comlpexity
    """
    N = args.num_points
    P = args.polynomial
    scaler = scale_conv[args.scaling]

    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, eps0=args.epsilon)

    results = defaultdict(lambda: np.zeros(len(P), dtype=float))
    resamp = resampling_conv[args.resampling]

    for i, p in enumerate(P):
        print("p = ", p)
        X = utils.create_X(x, y, p)

        data = resamp(X, z, args.tts, args.resampling_iter,  args.lmb[0], reg_conv[args.method], scaler)

        results["test_errors"][i] = data["test_MSE"]
        results["test_biases"][i] = data["test_bias"]
        results["test_vars"][i] = data["test_variance"]

        results["train_errors"][i] = data["train_MSE"]
        results["train_biases"][i] = data["train_bias"]
        results["train_vars"][i] = data["train_variance"]

    plot.Plot_bias_var_tradeoff(results, args)


def lambda_BVT(args):
    """
    Perform bias-variance trade-off analysis for different 
    values of lambda, as per last paragraph of Ex4

    Should be used with Ridge or Lasso as regression methods
    Should be used with Bootstrapping or CV as resampling methods

    Plots MSE for test as function of complexity and lambda-parameter
    """
    N = args.num_points
    P = args.polynomial
    lmbs = args.lmb
    scaler = scale_conv[args.scaling]

    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, eps0=args.epsilon)

    results = defaultdict(lambda: np.zeros((len(P), len(lmbs)), dtype=float))
    resamp = resampling_conv[args.resampling]

    for i, p in enumerate(P):
        print("p = ", p)
        X = utils.create_X(x, y, p)

        for k, lmb in enumerate(lmbs):
            print("    lmb = ", lmb)
            data = resamp(X, z, args.tts, args.resampling_iter, lmb, reg_conv[args.method], scaler)

            results["test_MSE"][i][k] = data["test_MSE"]
    
    plot.Plot_BVT_lambda(results, args)
