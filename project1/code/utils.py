import argparse
import numpy as np


def parameter_range(inp, method, lmb=False):
    """
    For polynomial degree and lambda parameter.
    Returns a string of a single value, a list, or an arange/logspace
    Must be evaluated later. This is done in main.
    """
    if lmb:
        ntype = float
        func = "np.logspace"
    else:
        ntype = int
        func = "np.arange"

    if method == "value":
        return f"({ntype(inp)},)"  # tuple with single element
    elif method == "list":
        return str(sorted([ntype(i) for i in inp.split(",")]))

    return f"{func}({inp})"


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Explore different regression methods'
                    + 'and evaluate which one is best.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arg = parser.add_argument

    add_arg('-m', '--method',
            type=str,
            default='OLS',
            choices=['OLS', 'Ridge', 'Lasso'],
            help='Choose which regression method to use.',
            )

    add_arg("-tts", "--tts",
            type=float,
            default=0.2,
            choices=[0.2, 0.25, 0.3, 0.4],
            help="Train/test split ratio"
            )

    add_arg('-p', '--polynomial',
            type=str,
            default="3",
            help='Polynomial degree.',
            )

    add_arg('-pc', '--polynomial-conversion',
            type=str,
            default="value",
            choices=["value", "list", "range"],
            help="How to transform polynomial input",
            )

    add_arg('-n', '--num_points',
            type=int,
            default=100,
            help='Number of gridpoints along 1 axis',
            )

    add_arg('-s', '--scaling',
            type=str,
            default='S',
            choices=['None', 'M', 'S', 'N'],
            help='Scaling method: None, MinMax, Standard, Normalizer.',
            )

    add_arg('-r', '--resampling',
            type=str,
            default='None',
            choices=['None', 'Boot', 'CV'],
            help='Resamplingmethod: NoResampling, Bootstrap, Cross_Validation.',
            )

    add_arg("-ri", "--resampling-iter",
            type=int,
            default=None,
            help="Unused for NoResampling, B for Bootstrap, k.fold for CV",
            )

    add_arg('-l', '--lmb',
            type=str,
            default="0",
            help='Lambda parameter',
            )

    add_arg('-lc', '--lambda-conversion',
            type=str,
            default='value',
            choices=['value', 'list', 'range'],
            help='How to transform lambda input.',
            )

    add_arg("-d", "--dataset",
            type=str,
            default="Franke",
            choices=["Franke", "SRTM"],
            help="Dataset to be used. If SRTM, -df must give path to file "
            )

    add_arg("-df", "--data-file",
            type=str,
            default=None,
            help="Path to SRTM data file",
            )

    add_arg("-e", "--epsilon",
            type=float,
            default=0,
            help="Scale value of noice for Franke Function",
            )

    args = parser.parse_args(args)

    args.polynomial = parameter_range(args.polynomial, args.polynomial_conversion)
    args.lmb = parameter_range(args.lmb, args.lambda_conversion, True)

    print("Runtime arguments:", args, "\n")

    args.polynomial = eval(args.polynomial)
    args.lmb = eval(args.lmb)

    return args


def FrankeFunction(x, y, eps0=0):
    """
    Franke Function with noise.

    Parameters:
        x, y: array-like
            x,y-values
        eps0: flaot
            scale value for noise. Defaults to 0
    Returns:
        z: ndarray
            z-values
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = eps0 * np.random.normal(size=x.shape)
    return term1 + term2 + term3 + term4 + noise


def create_X(x, y, n):
    """
    Sets up design matrix

    Parameters:
        x, y: array-like
            Are flattened if not already
        n: int
            max polynomial degree
    Returns:
        X: 2darray
            Includes intercept.
    """
    if not 1 in x.shape:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = (n + 1) * (n + 2) // 2  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = i * (i + 1) // 2
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)
    return X


def MSE(y, y_predict):
    return sum((y - y_predict) ** 2) / len(y)


def R2(y, y_predict):
    return 1 - sum((y - y_predict) ** 2) / sum((y - np.mean(y)) ** 2)
