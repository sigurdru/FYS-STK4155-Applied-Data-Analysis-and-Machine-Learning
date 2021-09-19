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
        return str(ntype(inp))
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
            default='None',
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
            help="Unused for NoResampling, B for Bootstrap, k for CV",
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
            help="Path to SRTM data file")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    args.polynomial = parameter_range(args.polynomial, args.polynomial_conversion)
    args.lmb = parameter_range(args.lmb, args.lambda_conversion, True)

    print("Runtime arguments:", args)

    args.polynomial = eval(args.polynomial)
    args.lmb = eval(args.lmb)

    return args
