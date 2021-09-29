import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import resample
import utils
import regression
import resampling
import plot


class NoneScaler(StandardScaler):
    """ To have the option of no scaling of the data """
    def transform(self, X):
        return X

reg_conv = {"OLS": regression.Ordinary_least_squares, "Ridge": regression.Ridge, "Lasso":regression.Lasso}
resampling_conv = {"None": resampling.NoResampling, "Boot": resampling.Bootstrap, "CV": resampling.cross_validation}
scale_conv = {"None": NoneScaler(), "S": StandardScaler(), "N": Normalizer(), "M": MinMaxScaler()}


def tmp_func_name(args):
    N = args.num_points
    P = args.polynomial  # polynomial degrees
    scaler = scale_conv[args.scaling]

    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, eps0=args.epsilon)
    MSEs = np.zeros(len(P))
    MSE_train = np.zeros(len(P))

    for i, p in enumerate(P):
        print("p =", p)
        X = utils.create_X(x, y, p)

        resampl = resampling_conv[args.resampling]

        data = resampl(X, z, args.tts, args.resampling_iter, args.lmb, reg_conv[args.method], scaler)
        MSEs[i] = data["test_MSE"]
        MSE_train[i] = data["train_MSE"]
    #Plotting the error, see output folder!
    plot.Plot_error(pol_deg=P, MSE_test=MSEs, MSE_train=MSE_train, args=args)

def bias_var_tradeoff(args):
    N = args.num_points
    P = args.polynomial
    scaler = scale_conv[args.scaling]

    x = np.sort(np.random.normal(size=N))
    y = np.sort(np.random.normal(size=N))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, eps0=args.epsilon)

    test_errors = np.zeros(len(P), dtype=float)
    test_biases = np.zeros(len(P), dtype=float)
    test_vars = np.zeros(len(P), dtype=float)

    train_errors = np.zeros(len(P), dtype=float)
    train_biases = np.zeros(len(P), dtype=float)
    train_vars = np.zeros(len(P), dtype=float)

    for i, p in enumerate(P):
        print("p = ", p)
        X = utils.create_X(x, y, p)

        resamp = resampling_conv[args.resampling]

        data = resamp(X, z, args.tts, args.resampling_iter,  args.lmb, reg_conv[args.method], scaler)

        test_errors[i] = data["test_MSE"]
        test_biases[i] = data["test_bias"]
        test_vars[i] = data["test_variance"]

        train_errors[i] = data["train_MSE"]
        train_biases[i] = data["train_bias"]
        train_vars[i] = data["train_variance"]

    plt.plot(P, test_errors, "bo-", label="test Error")
    plt.plot(P, test_biases, "ro-", label="test Bias")
    plt.plot(P, test_vars, "go-", label="test Variance")
    plt.plot(P, train_errors, "bo--", label="train Error")
    plt.plot(P, train_biases, "ro--", label="train Bias")
    plt.plot(P, train_vars, "go--", label="train Variance")
    plt.legend()
    plt.show()