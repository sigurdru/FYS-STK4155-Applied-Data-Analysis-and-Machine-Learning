import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')
here = os.path.abspath(".")
path_plots = '../output_extra/plots/'

def show_save(fig, fname, args):
    """
    This function handles wether you want to show and/or save the file.

    Args:
        fig (matplotlib.figure): Figure you want to handle
        fname (string):  filename
        args (argparse)
    """
    file = path_plots + fname + '.pdf'
    if args.save:
        print(f'Saving plot: file://{here}/{file}')
        fig.savefig(file)
    if args.show:
        plt.show()
    else:
        plt.clf()
    print("\n\n")

def set_ax_info(ax, xlabel, ylabel, style=None, title=None):
    """Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        title (str): the desired title on the axis
        xlabel (str): the desired lab on the x-axis
        ylabel (str): the desired lab on the y-axis
    """
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    if style is not None:
        ax.ticklabel_format(style=style)
    ax.legend(fontsize=15)


def bias_var_linreg(mses, biases, vars, fig, ax, args):
    """Plotting function for linear regression methods.
    
    Args:
        mses (dict): MSE for each complexity
        biases (dict): bias for each complexity
        vars (dict): variance for each complexity
        fig: matplotlib figure instance
        ax: matplotlib ax instance
        args (argparse): arguments from argparse
        
    """
    xaxis = range(1, args.nr_complexity+1)

    ax.plot(xaxis, mses.values(), c='b', label='mse')
    ax.plot(xaxis, biases.values(), c='g', label='bias')
    ax.plot(xaxis, vars.values(), c='r', label='variance')

    xlabel = 'polynomial degree'
    ylabel = 'error'
    title = f'Bias-variance tradeoff for {args.method}'
    fname = f'bias_var_{args.method}_c{args.nr_complexity}'

    set_ax_info(ax, xlabel, ylabel, style='sci', title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

def bias_var_regularization(biases, vars, fig, ax, args):
    """Plotting function for Ridge and Lasso including regularization.

    Args:
        biases (dict): biases for each complexity
        vars (dict): variances for each complexity
        fig: matplotlib figure instance
        ax: matplotlib ax instance
        args (argparse): arguments from argparse

    """
    xaxis = range(1, args.nr_complexity+1)
    clrs_map = {0.001:'b', 0.01:'r', 0.1:'g', 1:'orange', 10:'black'}
    ax.plot(xaxis, biases.values(), c=clrs_map[args.regularization], \
            label=r'bias $\alpha$={}'.format(args.regularization))
    ax.plot(xaxis, vars.values(), '--', c=clrs_map[args.regularization], \
            label=r'variance $\alpha$={}'.format(args.regularization))

    if args.regularization == 10:
        xlabel = 'polynomial degree'
        ylabel = 'error'
        title = f'Bias-variance tradeoff for {args.method} for different regularizations'
        fname = f'bias_var_{args.method}_c{args.nr_complexity}_alpha'

        set_ax_info(ax, xlabel, ylabel, style='sci', title=title)
        fig.set_tight_layout(True)
        show_save(fig, fname, args)


def bias_var_NN(mses, biases, vars, fig, ax, args):
    """Plotting function for Neural Network.

    Args:
        mses (dict): MSEs for each complexity
        biases (dict): biases for each complexity
        vars (dict): variances for each complexity
        fig: matplotlib figure instance
        ax: matplotlib ax instance
        args (argparse): arguments from argparse
        
    """
    xaxis = [len(h) for h in args.nr_hidden_layers_nodes]

    ax.plot(xaxis, mses.values(), '--o', c='b', label='mse')
    ax.plot(xaxis, biases.values(), '--o', c='g', label='bias')
    ax.plot(xaxis, vars.values(), '--o', c='r', label='variance')

    xlabel = 'number of hidden layers'
    ylabel = 'error'
    title = 'Bias-variance tradeoff for Neural Network, \n'
    title += f'{args.nr_hidden_layers_nodes[0][0]} nodes in each hidden layer'
    fname = f'bias_var_NN_layers{len(args.nr_hidden_layers_nodes)}_nodes{args.nr_hidden_layers_nodes[0][0]}'

    set_ax_info(ax, xlabel, ylabel, title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)


def bias_var_svm(mses, biases, vars, fig, ax, args):
    """Plotting function for Support Vector Machine.

    Args:
        mses (dict): MSEs for each complexity
        biases (dict): biases for each complexity
        vars (dict): variances for each complexity
        fig: matplotlib figure instance
        ax: matplotlib ax instance
        args (argparse): arguments from argparse
        
    """
    xaxis = np.array(args.C_regularization).astype(float)

    ax.semilogx(xaxis, mses.values(), c='b', label='mse')
    ax.semilogx(xaxis, biases.values(), c='g', label='bias')
    ax.semilogx(xaxis, vars.values(), c='r', label='variance')

    xlabel = 'C'
    ylabel = 'error'
    title = f'Bias-variance tradeoff for Support Vector Machine'
    fname = f'bias_var_SVM_C{len(args.C_regularization)}'

    set_ax_info(ax, xlabel, ylabel, title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

def bias_var_svm_3D(biases, vars, args):
    """3D surface plot over regularization parameters C
    and epsilon for Support Vector Machine.
    
    Args:
        biases (dict): biases for each complexity
        vars (dict): variances for each complexity
        args (argparse): arguments from argparse
        
    """
    fig_bias, ax_bias = plt.subplots(subplot_kw={"projection": "3d"})
    fig_var, ax_var = plt.subplots(subplot_kw={"projection": "3d"})

    xaxis = np.array(args.C_regularization).astype(float)
    yaxis = np.array(args.epsilon).astype(float)
    xmesh, ymesh = np.meshgrid(xaxis, yaxis)
    xmesh, ymesh = xmesh.T, ymesh.T # transpose to match with bias/var values

    from matplotlib import cm
    xmesh_log = np.log10(xmesh)
    ymesh_log = np.log10(ymesh)
    biases_3d = np.array(list(biases.values())).reshape(xmesh_log.shape[0], ymesh.shape[1])
    vars_3d = np.array(list(vars.values())).reshape(xmesh_log.shape[0], ymesh.shape[1])

    surf_bias = ax_bias.plot_surface(xmesh_log, ymesh_log, biases_3d, cmap=cm.coolwarm, alpha=0.6)
    surf_var = ax_var.plot_surface(xmesh_log, ymesh_log, vars_3d, cmap=cm.viridis, alpha=0.6)

    xlabel = r'C ($10^x$)'
    ylabel = r'$\epsilon$ ($10^x$)'
    title_bias = f'Bias for Support Vector Machine'
    title_var = f'Variance for Support Vector Machine'

    fname_bias = f'bias_SVM_C{len(args.C_regularization)}_eps{len(args.epsilon)}'
    fname_var = f'var_SVM_C{len(args.C_regularization)}_eps{len(args.epsilon)}'
    fig_bias.colorbar(surf_bias, shrink=0.5, aspect=5)
    fig_var.colorbar(surf_var, shrink=0.5, aspect=5)
    ax_var.view_init(25, 130)

    set_ax_info(ax_bias, xlabel, ylabel, style='sci', title=title_bias)
    set_ax_info(ax_var, xlabel, ylabel, style='sci', title=title_var)
    show_save(fig_bias, fname_bias, args)
    show_save(fig_var, fname_var, args)