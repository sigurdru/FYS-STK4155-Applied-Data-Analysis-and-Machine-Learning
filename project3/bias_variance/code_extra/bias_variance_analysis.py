from bootstrap import Bootstrap, FrankeFunction
from bias_variance_utils import *
import bias_variance_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline, make_pipeline

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def run_linear_regression(bs_model, fig, ax, args):
    """Runs bias-variance calculations for linear regression,
    including plotting.
    
    Args:
        bs_model (Bootstrap): model from Bootstrap class
        fig: matplotlib figure instance
        ax: matplotlib axis instance
        
    Returns:
        mse_all (array): mse for all complexities
        bias_all (array): bias for all complexities
        var_all (array): variance for all complexities
        
    """
    mse_all = {}
    bias_all = {}
    var_all = {}

    if args.method == 'OLS':
        linreg_model = LinearRegression(fit_intercept=False)
    elif args.method == 'Ridge':
        linreg_model = Ridge(fit_intercept=False, alpha=args.regularization)
    elif args.method == 'Lasso':
        linreg_model = Lasso(fit_intercept=False, alpha=args.regularization)

    for d in range(1, args.nr_complexity+1):
        pipe_model = make_pipeline(PolynomialFeatures(degree=d), linreg_model)

        t_tilde, t_pred = bs_model.simulate(pipe_model)
        mse, bias, var = bs_model.mse_decomposition(t_pred)

        mse_all[d] = mse
        bias_all[d] = bias
        var_all[d] = var

    if args.method == 'OLS':
        bias_variance_plot.bias_var_linreg(mse_all, bias_all, var_all, fig, ax, args)

    elif args.method in ['Ridge', 'Lasso']:

        if args.plot_type == 'regularization':
            bias_variance_plot.bias_var_regularization(bias_all, var_all, fig, ax, args)
            args.regularization *= 10
            mses, biases, vars = run_linear_regression(bs_model, fig, ax, args)
        
        elif args.plot_type == 'standard':
            bias_variance_plot.bias_var_linreg(mse_all, bias_all, var_all, fig, ax, args)


    return mse_all, bias_all, var_all


def run_neural_network(bs_model, fig, ax, args):
    """Runs bias-variance calculations for neural network,
    including plotting.
    
    Args:
        bs_model (Bootstrap): model from Bootstrap class
        fig: matplotlib figure instance
        ax: matplotlib axis instance
        
    Returns:
        mse_all (array): mse for all complexities
        bias_all (array): bias for all complexities
        var_all (array): variance for all complexities
        
    """
    mse_all = {}
    bias_all = {}
    var_all = {}

    for h in args.nr_hidden_layers_nodes:
        model_nn = MLPRegressor(hidden_layer_sizes=h, alpha=args.regularization, learning_rate_init=0.01)

        t_tilde, t_pred = bs_model.simulate(model_nn)
        mse, bias, var = bs_model.mse_decomposition(t_pred)

        mse_all[h] = mse
        bias_all[h] = bias
        var_all[h] = var 

    if args.plot_type == 'standard':
        bias_variance_plot.bias_var_NN(mse_all, bias_all, var_all, fig, ax, args)

    return mse_all, bias_all, var_all

    
def run_support_vector_machine(bs_model, fig, ax, args):
    """Runs bias-variance calculations for support vector machine,
    including plotting.
    
    Args:
        bs_model (Bootstrap): model from Bootstrap class
        fig: matplotlib figure instance
        ax: matplotlib axis instance
        
    Returns:
        mse_all (array): mse for all complexities
        bias_all (array): bias for all complexities
        var_all (array): variance for all complexities
        
    """
    Cs = np.array(args.C_regularization).astype(float)
    epsilons = np.array(args.epsilon).astype(float)

    mse_all = {}
    bias_all = {}
    var_all = {}
    
    for c in Cs:
        for e in epsilons:
            svm_model = SVR(kernel=args.kernel, C=c, epsilon=e)

            t_tilde, t_pred = bs_model.simulate(svm_model)
            mse, bias, var = bs_model.mse_decomposition(t_pred)

            mse_all[(c,e)] = mse
            bias_all[(c,e)] = bias
            var_all[(c,e)] = var

    if args.plot_type == 'standard':
        bias_variance_plot.bias_var_svm(mse_all, bias_all, var_all, fig, ax, args)

    elif args.plot_type == '3d': 
        bias_variance_plot.bias_var_svm_3D(bias_all, var_all, args)


    return mse_all, bias_all, var_all