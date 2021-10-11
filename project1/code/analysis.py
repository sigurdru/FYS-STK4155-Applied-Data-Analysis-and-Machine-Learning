import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split as tts
from regression import Ordinary_least_squares, Ridge, Lasso
from resampling import NoResampling, Bootstrap, cross_validation
import utils
import plot


class NoneScaler(StandardScaler):
    """ To have option of no scaling """
    def transform(self, x):
        return x

# Dicts converting from string to callable functions
reg_conv = {"OLS": Ordinary_least_squares, "Ridge": Ridge, "Lasso":Lasso}
resampling_conv = {"None": NoResampling, "Bootstrap": Bootstrap, "CV": cross_validation}
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


def simple_regression(args):
    """
    Run regression. Default analysis function
    For Ridge and Lasso, a single lambda-value is used

    Can be used with any regression method
    Can be used with any resampling method

    Plot MSE for train and test as function of complexity
    """
    P = args.polynomial  # polynomial degrees
    scaler = scale_conv[args.scaling]

    x, y, z = utils.load_data(args)

    # plot.Plot_3DDataset(x, y, z, args)

    MSEs = np.zeros(len(P))
    MSE_train = np.zeros(len(P))
    R2s = np.zeros(len(P))
    R2_train = np.zeros(len(P))

    resampl = resampling_conv[args.resampling]

    X = utils.create_X(x, y, P[-1])
    X_train_, X_test_, z_train, z_test = split_scale(X, z, args.tts, scaler)

    for i, p in enumerate(P):
        print("p = ", p)

        if args.resampling != "CV":
            X_train = X_train_[:, :utils.get_features(p)]
            X_test = X_test_[:, :utils.get_features(p)]
            inputs = ((X_train, X_test), (z_train, z_test), args.resampling_iter, args.lmb[0], reg_conv[args.method])
        else:
            inputs = (X[:, :utils.get_features(p)], z, args.resampling_iter, args.lmb[0], reg_conv[args.method])
        data = resampl(*inputs)

        MSEs[i] = data["test_MSE"]
        MSE_train[i] = data["train_MSE"]
        R2s[i] = data["test_R2"]
        R2_train[i] = data["train_R2"]

    # Plotting the error, see output folder!
    plot.Plot_error(MSE_test=MSEs, MSE_train=MSE_train, args=args)
    plot.Plot_R2(R2_test=R2s, R2_train=R2_train, args=args)

    if args.pred:
        X_ = X[:, :utils.get_features(p)]
        beta = reg_conv[args.method](X_,z)
        plot.Plot_3DDataset(x, y, X_ @ beta, args, predict=True)

    if args.method == "OLS" and args.dataset == "Franke" and not args.show:
        """ For Ex1 we want to make a plot of the variance in the beta values. """
        plot.Plot_VarOLS(args)


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
    resampl = resampling_conv[args.resampling]

    X = utils.create_X(x, y, P[-1])
    X_train_, X_test_, z_train, z_test = split_scale(X, z, args.tts, scaler)

    for i, p in enumerate(P):
        print("p = ", p)

        if args.resampling != "CV":
            X_train = X_train_[:, :utils.get_features(p)]
            X_test = X_test_[:, :utils.get_features(p)]
            inputs = ((X_train, X_test), (z_train, z_test), args.resampling_iter, args.lmb[0], reg_conv[args.method])
        else:
            inputs = (X[:, :utils.get_features(p)], z, args.resampling_iter, args.lmb[0], reg_conv[args.method])
        data = resampl(*inputs)

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


def lambda_analysis(args):
    """
    Performs lambda analysis

    Should be used with Ridge and Lasso as regression methods
    Should be used with bootstrapping or CV as resamling method

    If not too many polydegree:
    Plots test MSE as function of lambda. Can be done for multiple polynomial degrees
    If too many polydegree:
    Plots a contour of MSE as function of complexity and lambda
    """
    P = args.polynomial
    lmbs = args.lmb
    scaler = scale_conv[args.scaling]

    x, y, z = utils.load_data(args)

    results = defaultdict(lambda: np.zeros((len(P), len(lmbs)), dtype=float))
    resampl = resampling_conv[args.resampling]

    X = utils.create_X(x, y, P[-1])
    X_train_, X_test_, z_train, z_test = split_scale(X, z, args.tts, scaler)

    for i, p in enumerate(P):
        print("p = ", p)

        if args.resampling != "CV":
            X_train = X_train_[:, :utils.get_features(p)]
            X_test = X_test_[:, :utils.get_features(p)]
        else:
            X_ = X[:, :utils.get_features(p)]

        for k, lmb in enumerate(lmbs):
            print("    l = ", lmb)
            if args.resampling != "CV":
                inputs = ((X_train, X_test), (z_train, z_test), args.resampling_iter, lmb, reg_conv[args.method])
            else:
                inputs = (X_, z, args.resampling_iter, lmb, reg_conv[args.method])
            data = resampl(*inputs)

            results["test_MSE"][i][k] = data["test_MSE"]

    r = results["test_MSE"]
    print(np.where(r == np.min(r)))
    print(np.min(r))

    if len(P) > 5:
        plot.Plot_2D_MSE(results, args)
    else:
        plot.Plot_lambda(results, args)


def BVT_lambda(args):
    """
    Perform bias-variance trade-off analysis for different
    values of lambda, as per last paragraph of Ex4

    Should be used with Ridge or Lasso as regression methods
    Should be used with Bootstrapping resampling methods

    Plots MSE for test as function of complexity for different lambda-parameter
    """
    P = args.polynomial
    lmbs = args.lmb
    scaler = scale_conv[args.scaling]

    x, y, z = utils.load_data(args)

    results = defaultdict(lambda: np.zeros((len(P), len(lmbs)), dtype=float))
    resampl = resampling_conv[args.resampling]  # Should be bootstrapping

    X = utils.create_X(x, y, P[-1])
    X_train_, X_test_, z_train, z_test = split_scale(X, z, args.tts, scaler)

    for i, p in enumerate(P):
        print("p = ", p)

        if args.resampling != "CV":
            X_train = X_train_[:, :utils.get_features(p)]
            X_test = X_test_[:, :utils.get_features(p)]
        else:
            X_ = X[:, :utils.get_features(p)]

        for k, lmb in enumerate(lmbs):
            print("    l = ", lmb)
            if args.resampling != "CV":
                inputs = ((X_train, X_test), (z_train, z_test), args.resampling_iter, lmb, reg_conv[args.method])
            else:
                inputs = (X_, z, args.resampling_iter, lmb, reg_conv[args.method])
            data = resampl(*inputs)

            results["test_errors"][i][k] = data["test_MSE"]
            results["test_biases"][i][k] = data["test_bias"]
            results["test_vars"][i][k] = data["test_variance"]

    plot.Plot_BVT_lambda(results, args)
