import argparse
import numpy as np
import matplotlib.pyplot as plt
import bias_variance_analysis
from bias_variance_utils import *
from bootstrap import Bootstrap, FrankeFunction


def parse_args(args=None):
    """
    Uses argparse module to return an object containing
    all runtime arguments specified in command line
    """
    parser = argparse.ArgumentParser(
        description='Bias-variance tradeoff',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arg = parser.add_argument

    add_arg('-m', '--method',
            type=str,
            default='OLS',
            choices=['OLS', 'Ridge', 'Lasso', 'NN', 'SVM'],
            help='Choose which machine learning method to study.',
            )

    add_arg('-nr_c', '--nr_complexity',
            type=int,
            default=12,
            help='Number of complexity features to simulate for.\
                    Used for linear regression and neural network.')

    add_arg('-plot_type',
                type=str,
                default='standard',
                choices=['standard', 'regularization', '3d'],
                help='Type of bias-variance plot.')    

    add_arg('-alpha', '--regularization',
            type=float,
            default=0.0001,
            help='Regularization parameter.')

    add_arg('-hidden', '--nr_hidden_layers_nodes',
            type=list,
            default=[(10,), (10,)*3, (10,)*5, (10,)*7, (10,)*9, (10,)*11, (10,)*13],
            help='Number of hidden layers and nodes for NN. List of tuples.')

    add_arg('-kernel', 
                type=str,
                default='poly',
                choices=['poly', 'linear', 'rbf'],
                help='Kernel to use for support vector machine.')

    add_arg('-C', '--C_regularization',
                nargs='+',
                help='Regularization term for SVM that determines the extent of the margin.\
                        Larger => less regularization => higher complexity.')

    add_arg('-eps', '--epsilon',
                nargs='+',
                help='Threshold value for ignoring residuals in optimization.\
                        Larger => more bias (more residuals ignored)')

    add_arg("-show",
            action="store_true",
            dest="show",
            )

    add_arg("-nosave",
            action="store_false",
            dest="save",
            )

    add_arg("-seed",
            type=int,
            default=1999,
            help="Random seed. If 0, no seed is used",
            )
    args = parser.parse_args(args)
    print("Runtime arguments:", args, "\n")
    return args


def main():
    args = parse_args()

    # Data
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y, noise_std=0.2, add_noise=True)

    if args.seed:
        np.random.seed(args.seed)

    boot = Bootstrap(x,y,z, args, scale_target=True)

    fig, ax = plt.subplots(figsize=[10, 5])

    if args.method in ['OLS', 'Ridge', 'Lasso']:
        mse_linreg, bias_linreg, var_linreg = \
            bias_variance_analysis.run_linear_regression(boot, fig, ax, args)

    elif args.method == 'NN':
        mse_NN, bias_NN, var_NN = \
            bias_variance_analysis.run_neural_network(boot, fig, ax, args)

    elif args.method == 'SVM':
        mse_svm, bias_svm, var_svm = \
            bias_variance_analysis.run_support_vector_machine(boot, fig, ax, args)


if __name__ == "__main__":
    main()