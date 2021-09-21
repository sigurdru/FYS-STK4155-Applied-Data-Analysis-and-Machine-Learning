import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import resample
import utils
import regression
import resampling


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
    z = utils.FrankeFunction(x, y, eps0=args.epsilon).flatten()
    MSEs = np.zeros(len(P))
    MSE_train = np.zeros(len(P))

    for i, p in enumerate(P):
        print("p =", p)
        X = utils.create_X(x, y, p)
        
        resampl = resampling_conv[args.resampling]

        data = resampl(X, z, args.tts, args.resampling_iter, args.lmb, reg_conv[args.method], scaler)
        MSEs[i] = data["test_MSE"] 
        MSE_train[i] = data["train_MSE"] 

    plt.plot(P, MSEs, "bo--", label="test MSE")
    plt.plot(P, MSE_train, "ro--", label="Train MSE")
    plt.legend()
    plt.show()


def bias_var_tradeoff(args):
    N = args.num_points
    P = args.polynomial
    scaler = scale_conv[args.scaling]

    x = np.sort(np.random.normal(size=N))
    y = np.sort(np.random.normal(size=N))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, eps0=args.epsilon).ravel().reshape(-1, 1)

    errors = np.zeros(len(P))
    biases = np.zeros(len(P))
    vars = np.zeros(len(P))

    for i, p in enumerate(P):
        print("p = ", p)
        X = utils.create_X(x, y, p)

        resamp = resampling_conv[args.resampling]

        data = resamp(X, z, args.tts, args.resampling_iter,  args.lmb, reg_conv[args.method], scaler)
        
        errors[i] = data["error"]
        biases[i] = data["bias"]
        vars[i] = data["variance"]

    plt.plot(P, errors, "bo--", label="Error")
    plt.plot(P, biases, "ro--", label="Bias")
    plt.plot(P, vars, "go--", label="Variance")
    plt.legend()
    plt.show()