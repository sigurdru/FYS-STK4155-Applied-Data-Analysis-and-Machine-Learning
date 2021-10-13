from matplotlib import transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from random import random, seed
import utils

# to suppress warnings from fig.tight_layout() in some plotsFalse
# import warnings
# warnings.filterwarnings("ignore")

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')
path_plots = '../output/plots'


def set_ax_info(ax, xlabel, ylabel, title=None, zlabel=None):
    """Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        title (str): the desired title on the axis
        xlabel (str): the desired lab on the x-axis
        ylabel (str): the desired lab on the y-axis
        zlabel (str): the desired lab on the z-axis

    """
    if zlabel is None:
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.ticklabel_format(style='plain')
        ax.legend(fontsize=15)
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_zlabel(zlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)

def push(file):
    os.system(f"git add {file}")
    os.system("git commit -m 'plots'")
    os.system("git push")
    print(f"Pushed to git: {file}")


def show(fig, fname, args):
    if args.dataset == "SRTM":
        print("Terrain data: \'SRTM_\' added at beginning of plot file name")
        fname = "SRTM_" + fname
    if args.save:
        print("Saving plot: ", fname + ".pdf")
        fig.savefig(os.path.join(path_plots, fname + '.pdf'))
    if args.push:
        push(os.path.join(path_plots, fname + ".pdf"))
    if args.show:
        plt.show()
    else:
        plt.clf()


def Plot_3DDataset(x, y, z, args, predict=False):
    """3D plot the data and saves the plot in the output folder
        Either Franke funcprint(cm.hot(0.3))tion or terrain data
    Args:
        x (array):  x-data
        y (array):  y-data
        z (array):  z-values
        args (argparse): argparse containing info about run
    """

    if 1 in z.shape:
        nx = x.shape[0]
        ny = x.shape[1]
        z = z.reshape((nx, ny))

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # general formalities
    if predict:
        fname = "prediction_p" + str(max(args.polynomial))
        title = f"Prediction of raw data for P = {max(args.polynomial)}"
    else:
        fname = f"rawdata"
        title = "Raw data"
    if args.dataset == "Franke":
        fname += f"_eps{args.epsilon}".replace(".", "")
    fname = fname.replace('.', '')  # remove dots from fname
    xlabel = '$x$'
    ylabel = '$y$'
    zlabel = '$z$'
    set_ax_info(ax, xlabel, ylabel, title, zlabel)
    fig.tight_layout()
    show(fig, fname, args)


def Plot_error(MSE_test, MSE_train, args):
    """Plot mean square error as a function of polynomial degree
    for test and train data

    Args:
        MSE_test (array): array of test mean square error
        MSE_train (array): array of train mean square error
        args (argparse): argparse containing information of method used
    """
    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(args.polynomial, MSE_test, "bo--", label="test MSE")
    ax.plot(args.polynomial, MSE_train, "ro--", label="Train MSE")

    amin = np.argmin(MSE_test)
    print(f"Lowest error is {MSE_test[amin]}, for p = {args.polynomial[amin]}")

    # general formalities
    if args.dataset == "SRTM":
        fname = 'MSE_' + args.method \
                + '_n' + str(args.num_points) \
                + '_pol' + str(max(args.polynomial))

        title = 'Terrain data MSE for ' + args.method

    else:
        fname = 'MSE_' + args.method \
                + '_n' + str(args.num_points) \
                + '_eps' + str(args.epsilon) \
                + '_pol' + str(max(args.polynomial))

        title = 'MSE for ' + args.method

    fname = fname.replace('.', '')  # remove dots from fname

    if args.resampling != "None":
        title += " using " \
                 + args.resampling \
                 + f" iter = {args.resampling_iter}"
        if args.method != "OLS":
            title += r', $\lambda$ = ' + str(args.lmb[0])
        fname += "_" + args.resampling + "_" + "re" \
                 + str(args.resampling_iter)
    if args.method != "OLS":
        fname += '_lam_'+str(args.lmb[0])
        fname = fname.replace('.', '_')
    xlabel = 'Polynomial degree'
    if args.log:
        ylabel = 'log(MSE)'
    else:
        ylabel = 'MSE'

    set_ax_info(ax, xlabel, ylabel, title)
    if args.log:
        ax.set_yscale('log')
        fname += '_log'

    # save figure
    print('Plotting error: See ' + fname + '.pdf')
    show(fig, fname, args)


def Plot_R2(R2_test, R2_train, args):
    """Plot R^2 score as a function of polynomial degree
    for test and train data

    Args:
        R2_test (array): array of test R2 score
        R2_train (array): array of train R2 score
        args (argparse): argparse containing information of method used
    """
    if args.resampling != "None":
        return
    elif args.method != "OLS":
        return
    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(args.polynomial, R2_test, "bo--", label="Test R2")
    ax.plot(args.polynomial, R2_train, "ro--", label="Train R2")
    # general formalitiesiscrete uniform”
    if args.dataset == "SRTM":
        fname = 'R2_' + args.method \
                + '_n' + str(args.num_points) \
                + '_eps' + str(args.epsilon) \
                + '_pol' + str(max(args.polynomial))
        title = "Terrain data R2 score for " \
                + args.method

    else:
        fname = 'R2_' + args.method \
                + '_n' + str(args.num_points) \
                + '_eps' + str(args.epsilon) \
                + '_pol' + str(max(args.polynomial))

        title = 'R2 score for ' \
                + args.method

    fname = fname.replace('.', '')  # remove dots from fname
    xlabel = 'Polynomial degree'
    ylabel = 'R2'
    set_ax_info(ax, xlabel, ylabel, title)

    # save figure
    print('Plotting R2: See ' + fname + '.pdf')
    show(fig, fname, args)


def Plot_bias_var_tradeoff(datas, args):
    """Plot mean square error, variance and error as a function of polynomial degree

    Args:
        datas (defaultdict) contains following arrays
                test_errors
                test_biases
                test_vars
                train_errors
                train_biases
                train_vars
        args (argparse): argparse containing information of method used
    """
    # make figure and plot data
    fig, ax = plt.subplots()

    P = args.polynomial
    ax.plot(P, datas["test_errors"], "bo-", label="test MSE")
    ax.plot(P, datas["test_biases"], "ro-", label="test Bias")
    ax.plot(P, datas["test_vars"], "go-", label="test Variance")
    ax.plot(P, datas["train_errors"], "bo--", label="train MSE")
    ax.plot(P, datas["train_biases"], "ro--", label="train Bias")
    ax.plot(P, datas["train_vars"], "go--", label="train Variance")
    # General formalities
    xlabel = 'Polynomial degree'
    ylabel = 'Bias, variance and MSE'
    title = 'Bias Variance Tradeoff for ' + args.method+', '\
        + args.resampling + ' iter = '+str(args.resampling_iter)
    if args.log:
        ylabel = '(log) Bias, variance and MSE'
    set_ax_info(ax, xlabel, ylabel, title)

    # Saving figure
    if args.dataset == "SRTM":
        fname = 'BVT_' + args.method \
                + '_n' + str(args.num_points)
        title = 'Terrain data: ' + title

    else:
        fname = 'BVT_' + args.method \
                + '_n' + str(args.num_points) \
                + '_eps' + str(args.epsilon)

    if args.method != "OLS":
        fname += '_lam_'+str(args.lmb[0])

    if args.log:
        ax.set_yscale('log')
        fname += '_log'

    fname = fname.replace('.', '_')
    print('Plotting Bias variance tradeoff: See %s.pdf' % (fname))
    show(fig, fname, args)


def Plot_lambda(results, args):
    """ Plot test MSE as function of lambda, for different polynomial degrees

    Args:
        results: (defaultdict) contains 1 2D array with test errors
                First index goes over poly-degree
                2nd index goes over lambda
    """
    fig, ax = plt.subplots()

    lmbs = args.lmb
    for i, p in enumerate(args.polynomial):
        r = results["test_MSE"][i]
        if args.log:
            r = np.log10(r)
        ax.plot(np.log10(lmbs), r, label=f"Polynomial degree: {p}")

    r = results["test_MSE"]
    min_val = np.min(r)
    mp, ml = np.where(r == min_val)
    mp = args.polynomial[mp[0]]
    ml = np.log10(lmbs[ml][0])
    print(f"Min MSE is {min_val} at p={mp}, l={ml}")


    xlabel = "log10(lambda-parameter)"
    ylabel = "log10(Error)" if args.log else "Error"
    title = f"Error using {args.method} and {args.resampling} iter = {args.resampling_iter}"

    if args.dataset == "SRTM":
        title = "Terrain data: " + title
    set_ax_info(ax, xlabel, ylabel, title)

    low = str(int(np.log10(args.lmb[0]))).replace("-", "m")
    high = str(int(np.log10(args.lmb[-1]))).replace("-", "m")
    fname = f"lambdaMSE_{args.method}_{args.resampling}{args.resampling_iter}_n{args.num_points}"
    fname += f"_eps{args.epsilon}_p{args.polynomial[-1]}_l{low}_{high}"
    fname = fname.replace(".", "")
    print('Plotting lambda_analysis: See %s.pdf' % (fname))
    show(fig, fname, args)


def Plot_VarOLS(args):
    """Here we plot the parameters for different polynomials
    with confidence intervals.
    """
    p = np.array([1, 3, 5])
    n = 30
    sigma_sq = 0.2
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, sigma_sq)
    X_ = utils.create_X(x, y, p[-1])
    for i in p:
        X = X_[:, :utils.get_features(i)]
        XTXinverse = np.linalg.pinv(X.T @ X)
        beta = XTXinverse @ X.T @ z
        variance_beta = sigma_sq*XTXinverse
        sigma_sq_beta = np.diag(variance_beta)
        upper = beta[:, 0] + 2*np.sqrt(sigma_sq_beta)
        lower = beta[:, 0] - 2*np.sqrt(sigma_sq_beta)
        beta_index = np.arange(0, len(beta), 1)
        fig, ax = plt.subplots()
        ax.scatter(beta_index, beta, label=r'$\beta$(index)')
        ax.fill_between(beta_index, lower, upper,
                        alpha=0.2, label=r'$\pm 2\sigma$')
        xlabel = r'index'
        ylabel = r'$\beta$'
        title = r'$\beta$-values with confidence interval for p = %i' % (i)
        set_ax_info(ax, xlabel, ylabel, title)
        fname = 'Var_OLS_poldeg_' + str(i)
        show(fig, fname, args)
        print('Plotting variance in beta: See ' + fname + '.pdf')


def Plot_BVT_lambda(result, args):
    """
    Plots BVT for as function of polynomial degree for different lambda
    """
    fig, ax = plt.subplots()

    P = args.polynomial

    # c = np.linspace(0, 1, len(args.lmb))
    c = [0.1, 0.3, 0.9]
    cmap = cm.jet
    for j, lmb in enumerate(args.lmb):
        ax.plot(P, result["test_errors"][:, j], c=cmap(
            c[j]), label=f"MSE, lambda: {lmb}")
        ax.plot(P, result["test_biases"][:, j],
                "--", c=cmap(c[j]), label="bias")
        ax.plot(P, result["test_vars"][:, j], "-.",
                c=cmap(c[j]), label="variance")

    xlabel = "Polynomial degree"
    ylabel = "Bias, variance and error"
    title = f"BV-tradeoff for {args.method} using {args.resampling} iter = {args.resampling_iter}"
    set_ax_info(ax, xlabel, ylabel, title)

    fname = f"LBVT_{args.method}_{args.resampling}_n{args.num_points}"
    fname += f"_eps{args.epsilon}_p{args.polynomial[-1]}"
    low = str(int(np.log10(args.lmb[0]))).replace("-", "m")
    high = str(int(np.log10(args.lmb[-1]))).replace("-", "m")
    fname += f"_lmb{low}_{high}"
    fname = fname.replace(".", "_")
    print("Plotting lambda BVT: see ", fname, ".pdf")

    fig.tight_layout()
    show(fig, fname, args)


def Plot_2D_MSE(results, args):
    """
    Plots contour map of MSE as function of polynomial degree and lambda
    """
    fig, ax = plt.subplots()
    P, lmb = np.meshgrid(args.polynomial, np.log10(args.lmb))
    MSE = results["test_MSE"].T
    if args.log:
        MSE = np.log10(MSE)
        title = "log10(MSE)"
    else:
        title = "MSE"

    min_val = np.min(MSE)
    min_l, min_P = np.where(MSE == min_val)
    min_P = args.polynomial[min_P[0]]
    min_l = np.log10(args.lmb[min_l[0]])
    print(f"Min MSE is {min_val} at p={min_P}, l={min_l}")

    F = ax.contourf(P, lmb, MSE, cmap="jet")
    norm = mpl.colors.Normalize(vmin=F.cvalues.min(), vmax=F.cvalues.max())
    bar = mpl.cm.ScalarMappable(norm=norm, cmap=F.cmap)
    bar.set_array([])
    cbar = fig.colorbar(bar, ticks=F.levels)
    cbar.set_label("log10(Error)" if args.log else "Error",
                   rotation=90, fontsize=20, position=(1, 0.5))

    ax.scatter(min_P, min_l, s=30, c="r", marker="x")

    xlabel = "Polynomial degree"
    ylabel = "log10(lambda)"
    title += f" for {args.method} using {args.resampling} iter = {args.resampling_iter}"
    set_ax_info(ax, xlabel, ylabel, title)

    fname = f"Contour_PL_{args.method}_{args.resampling}{args.resampling_iter}_n{args.num_points}_eps{args.epsilon}"
    ph = str(args.polynomial[-1])
    pl = str(args.polynomial[0])
    lh = str(int(np.log10(args.lmb[0]))).replace("-", "m")
    ll = str(int(np.log10(args.lmb[-1]))).replace("-", "m")
    fname += f"_p{pl}_{ph}_lmb{ll}_{lh}"

    show(fig, fname, args)


if __name__ == "__main__":
    """
    Here we can plot Franke function with or withot noise,
    simply change the self.epsilon value to the desired noise.
    Then run python3 plot.py, and it will generate a file with the appropriate
    name.
    """
    class Argparse:
        def __init__(self, eps=0):
            self.show = False
            self.epsilon = eps
            self.dataset = "Franke"

    N = 30
    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)
    args = Argparse()
    z = utils.FrankeFunction(x, y, eps=args.epsilon)
