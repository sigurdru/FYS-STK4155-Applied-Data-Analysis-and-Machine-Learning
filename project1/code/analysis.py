import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from regression import Ordinary_least_squares, Ridge, Lasso
from resampling import NoResampling, Bootstrap, cross_validation
import utils
import plot

reg_conv = {"OLS": Ordinary_least_squares, "Ridge": Ridge, "Lasso":Lasso}
resampling_conv = {"None": NoResampling, "Boot": Bootstrap, "CV": cross_validation}
scale_conv = {"S": StandardScaler(), "N": Normalizer(), "M": MinMaxScaler()}


def simple_regression(args):
    """
    Run regression. Basically Ex1
    For Ridge and Lasso, a single lambda-value is used

    Can be used with any regression method
    Can be used with any resampling method

    Plot MSE for train and test as function of complexity
    """
    P = args.polynomial  # polynomial degrees
    scaler = scale_conv[args.scaling]

    x, y, z = utils.load_data(args)

    plot.Plot_FrankeFunction(x, y, z, args)

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
        R2s[i] = data["test_R2"]
        R2_train[i] = data["train_R2"]

    # Plotting the error, see output folder!
    plot.Plot_error(MSE_test=MSEs, MSE_train=MSE_train, args=args)
    plot.Plot_R2(R2_test=R2s, R2_train=R2_train, args=args)


def bias_var_tradeoff(args, testing=False):
    """
    Perform bias-variance trade-off analysis
    For Ridge and Lasso, a single lambda-value is used

    Can be used with any regression method
    Should be used with Bootstrapping as resampling method

    Plots MSE, bias and variance for train and test as function of comlpexity
    """
    P = args.polynomial
    scaler = scale_conv[args.scaling]

    x, y, z = utils.load_data(args)

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
    if testing == False:
        plot.Plot_bias_var_tradeoff(results, args)
    else:
        return results


def lambda_BVT(args):
    """
    Perform bias-variance trade-off analysis for different 
    values of lambda, as per last paragraph of Ex4

    Should be used with Ridge or Lasso as regression methods
    Should be used with Bootstrapping or CV as resampling methods

    Plots MSE for test as function of complexity and lambda-parameter
    """
    P = args.polynomial
    lmbs = args.lmb
    scaler = scale_conv[args.scaling]

    x, y, z = utils.load_data(args)

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
