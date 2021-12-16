"""
In this file we perform all plotting in this project.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')
here = os.path.abspath(".")
path_plots = '../output/plots/'
#archive = path_plots + "archive.txt"
def u_exact(x, t): return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


# def show_push_save(fig, func, args):
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

def set_ax_info(ax, xlabel, ylabel, style='plain', title=None):
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
    ax.ticklabel_format(style=style)
    ax.legend(fontsize=15)

def Euler_solution(x, t, u, args):
    """Plot numerical solution of Forward Euler 
    for 1D diffusion equation at different time steps.
    
    Args:
        x (array): spatial domain
        t (array): time domain
        u (array): numerical solution at desired time steps
        args (argparse): Information handled by the argparser
        
    """
    fig, ax = plt.subplots()

    for n in u.keys():
        ax.plot(x, u[n], label=r'$t={}$'.format(t[n]))

    title = 'Forward-Euler Solution of 1D diffusion equation'
    xlabel = '$x$'
    ylabel = '$u(x,t)$'
    fname = 'num_sol_FE'
    set_ax_info(ax, xlabel, ylabel, title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

def MSE_FE(x, t, u, args):
    """Calculates and plots MSE of numerical solution
    of forward euler.
    
    Args:
        x (array): spatial domain
        t (array): time domain
        u (array): numerical solution
        args (argparse): Information handled by the argparser

    Returns:
        (array): MSE between analytical and numerical solution
        
    """
    fig, ax = plt.subplots()

    mse = [np.sum( (u[n, :] - u_exact(x, t[n]))**2 )/len(x) for n in range(len(t))]

    ax.plot(t, mse)

    title = r'MSE of forward euler.' + '\n'
    title += r'$\alpha={}$, '.format(args.stability_criterion) 
    title += r'$\Delta x$ = ${}$'.format(args.x_step)
    xlabel = 't'
    ylabel = 'Error'
    fname = r'MSE_FE_dx_{}'.format(str(args.x_step).replace('.', ''))

    set_ax_info(ax, xlabel, ylabel, style='sci', title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

    return np.array(mse)

def error_x(x, t, u, args):
    """Calculates and plots difference between numerical and
    analytical soluton for two given time steps.
    
    Args:
        x (array): spatial domain
        t (array): time domain
        u (array): numerical solution at desired time steps
        args (argparse): Information handled by the argparser
        
    """
    fig, ax = plt.subplots()

    t_n = [t[n] for n in u.keys()]
    error_x = [u[n] - u_exact(x, t[n]) for n in u.keys()]

    for i, e in enumerate(error_x):
        ax.plot(x, e, label=r't={:.1f}'.format(t_n[i]))
    
    title = 'Error between numerical and analytical solution' + '\n' 
    title += 'at two time levels, using $\Delta x={}$'.format(args.x_step)
    xlabel = 'x'
    ylabel = 'Absolute error'
    fname = r'error_FE_x_dx_{}'.format(str(args.x_step).replace('.', ''))

    set_ax_info(ax, xlabel, ylabel, style='sci', title=title)
    fig.set_tight_layout(True)
    show_save(fig, fname, args)

def testing_data(model, args):
    """2D plot of initial state of neural network.
    
    Args:
        model (tf.keras.Sequential): deep neural network model \
                            with provided layers and parameters.
        args (argparse): Information handled by the argparser.

    """
    t_0, x_0, u_0 = model.t_0, model.x_0, model.u_0
    t_b, x_b, u_b = model.t_b, model.x_b, model.u_b
    t_r, x_r = model.t_r, model.x_r
    fig, ax = plt.subplots()
    ax.scatter(t_0, x_0, cmap='rainbow', c=u_0, marker='X')
    ax.scatter(t_b, x_b, cmap='rainbow', c=u_b, marker='X')
    ax.scatter(t_r, x_r, c='b', marker='.', alpha=0.1)
    
    title = 'Points where we will train the network'
    xlabel = 't'
    ylabel = 'x'
    set_ax_info(ax, xlabel, ylabel, title=title)
    
    fname = 'training_points'
    show_save(fig, fname, args)

def NN_diffusion_solution(model, args):
    """3D surface plot of solution of diffusion equation
    obtained by neural network.

    Args:
        model (tf.keras.Sequential): deep neural network model \
                            with provided layers and parameters.
        args (argparse): Information handled by the argparser.

    """
    Nx = 20
    Nt = 2000
    tspace = np.linspace(model.lb[0], model.ub[0], Nx + 1)
    xspace = np.linspace(model.lb[1], model.ub[1], Nt + 1)
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(), X.flatten()]).T
    
    upred = model.model(Xgrid) 
    U = upred.numpy().reshape(Nt+1, Nx+1)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('t', fontsize=20)
    ax.set_zlabel('u', fontsize=20)
    ax.set_title('Output of PINN', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.plot_surface(X, T, U, cmap='rainbow')
    ax.view_init(azim=120)
    fname = 'NN_diffusion_solution'
    fname += '_Nn' + str(args.num_neurons_per_layer) + '_Nh' + str(args.num_hidden_layers)
    show_save(fig, fname, args)

def NN_diffusion_error(loss_hist, args):
    """Plots loss of neural network for
    each training iteration.

    Args:
        loss_hist (list): loss for each iteration
        args (argparse): Information handled by the argparser.
    
    """
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(np.log10(loss_hist))

    title = r'Error of neural N$_{layers} = %i$ N$_{nodes} = %i$' % (args.num_hidden_layers, args.num_neurons_per_layer)
    xlabel = 'iterations'
    ylabel = r'$\log_{10}$(cost)'
    set_ax_info(ax, xlabel, ylabel, title=title)

    fname = 'NN_diffusion_error'
    fname += '_Nn' + str(args.num_neurons_per_layer) + '_Nh' + str(args.num_hidden_layers)
    print(f'Error after last iteration: {loss_hist[-1]}')
    show_save(fig, fname, args)

def NN_diffusion_error_timesteps(model, args):
    """Plots difference between output of neural network and 
    analytical solution for time steps t=0.1 and t=0.5.
    
    Args:
        model (tf.keras.Sequential): deep neural network model \
                            with provided layers and parameters.
        args (argparse): Information handled by the argparser.

    """
    Nx = 100
    Nt = Nx
    t1 = 0.1
    t2 = 0.5
    t1a = np.ones(Nt)*t1
    t2a = np.ones(Nt)*t2
    xa = np.linspace(0, 1, Nx)
    X1 = np.vstack([t1a, xa]).T
    X2 = np.vstack([t2a, xa]).T
    
    upred1 = model.model(X1)
    upred2 = model.model(X2)
    uexact1 = u_exact(xa, t1).reshape(-1, 1)
    uexact2 = u_exact(xa, t2).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.plot(xa, upred1 - uexact1, label='t = %.2f' %(t1))
    ax.plot(xa, upred2 - uexact2, label='t = %.2f' %(t2))
    title = r'Difference between analytical and output for $\Delta$x = 0.01'
    xlabel = 'x'
    ylabel = r'$u_{p} - u_{a}$'
    set_ax_info(ax, xlabel, ylabel, title=title)
    
    fname = 'NN_difference'
    fname += '_Nt' + str(args.num_train_iter)
    print('MSE at timestep %.2f: %f'%(t1, np.mean(np.sum((upred1-uexact1)**2))))
    print('MSE at timestep %.2f: %f'%(t2, np.mean(np.sum((upred2-uexact2)**2))))
    show_save(fig, fname, args)

def plot_eig_dim3(w_np, eigvec_nn, eigvec_fe, eigval_nn, 
                eigval_fe, s, t, v, args):
    """Plots predicted eigenvector and corresponding rayleigh 
    quotients for a 3x3 real, symmetric matrix,
    for neural network and forward euler.

    Args:
        w_np (array): true eigenvector
        eigvec_nn (array): predicted eigenvector from neural network
        eigvec_fe (array): predicted eigenvector from forward euler
        eigval_nn (array): rayleigh quotient from neural network
        eigval_fe (array): rayleigh quotient from forward euler
        s (array): time domain
        t (array): reshaped time domain
        v (array): eigenvalues of matrix
        args (argparse): Information handled by the argparser.

    """
    fig, ax = plt.subplots()
    ax.axhline(w_np[0], color='b', ls=':', label=r'Numpy $v_1$')
    ax.axhline(w_np[1], color='g', ls=':', label=r'Numpy $v_2$')
    ax.axhline(w_np[2], color='r', ls=':', label=r'Numpy $v_3$')

    ax.plot(t, eigvec_fe[:, 0], color='b', ls='--',
            label=r'Euler $v_1$')
    ax.plot(t, eigvec_fe[:, 1], color='g', ls='--',
            label=r'Euler $v_2$')
    ax.plot(t, eigvec_fe[:, 2], color='r', ls='--',
            label=r'Euler $v_3$')

    ax.plot(s, eigvec_nn[:, 0], color='b', label=r'NN $v_1$')
    ax.plot(s, eigvec_nn[:, 1], color='g', label=r'NN $v_2$')
    ax.plot(s, eigvec_nn[:, 2], color='r', label=r'NN $v_3$')
    ax.set_ylabel('Components of vector, $v$')
    ax.set_xlabel('Time, $t$')
    ax.set_title('Prediction of eigenvector corresponding to \
                largest eigenvalue', fontsize=20)
    ax.legend(loc='lower center', fancybox=True, 
                borderaxespad=0, ncol=3)
    
    fname = r'eigvec_T%i_N%i' %(args.tot_time, args.N_t_points)
    show_save(fig, fname, args)

    # Plot eigenvalues
    fig, ax = plt.subplots()
    ax.axhline(np.max(v), color='red', ls='--')
    ax.plot(t, eigval_fe)
    ax.plot(s, eigval_nn)

    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Rayleigh Quotient, $r$')
    ax.set_title('Prediction $r_{\\mathrm{final}}$ of \
                largest eigenvalue $\\lambda_{\\mathrm{max}}$', fontsize=20)
    lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
    str(round(np.max(v), 5))
    lgd_euler = "Euler $r_{\\mathrm{final}} \\sim$ " + \
        str(round(eigval_fe[-1], 5))
    lgd_nn = "NN $r_{\\mathrm{final}} \\sim$ " + \
        str(round(eigval_nn.numpy()[-1], 5))
    ax.legend([lgd_numpy, lgd_euler, lgd_nn], loc='lower center',
            fancybox=True, borderaxespad=0, ncol=1)

    fname = r'eigval_T%i_N%i' %(args.tot_time, args.N_t_points)
    show_save(fig, fname, args)


def plot_eig_dim6(w_np, eigvec_nn, eigvec_fe, eigval_nn, eigval_fe, s, t, v, args):
    """Plots predicted eigenvector and corresponding rayleigh 
    quotients for a 6x6 real, symmetric matrix,
    for neural network and forward euler.

    Args:
        w_np (array): true eigenvector
        eigvec_nn (array): predicted eigenvector from neural network
        eigvec_fe (array): predicted eigenvector from forward euler
        eigval_nn (array): rayleigh quotient from neural network
        eigval_fe (array): rayleigh quotient from forward euler
        s (array): time domain
        t (array): reshaped time domain
        v (array): eigenvalues of matrix
        args (argparse): Information handled by the argparser.

    """
    # Plot Neural Network
    fig, ax = plt.subplots()
    clr = ['b', 'g', 'r', 'orange', 'purple', 'black']

    for i in range(6):
        ax.axhline(w_np[i], color=clr[i], ls=':', label=f'Numpy $v_{i+1}$')
        ax.plot(s, eigvec_nn[:, i], color=clr[i], label=f'NN $v_{i+1}$')

    ax.set_ylabel('Components of vector, $v$')
    ax.set_xlabel('Time, $t$')
    ax.set_title('Prediction of eigenvectors - Neural Network', fontsize=20)
    ax.legend(loc='lower center', fancybox=True, 
                borderaxespad=0, ncol=3)
    
    fname = r'NN_eigvec_T%i_N%i_dim6' %(args.tot_time, args.N_t_points)
    show_save(fig, fname, args)

    # Plot Forward Euler
    fig, ax = plt.subplots()
    clr = ['b', 'g', 'r', 'orange', 'purple', 'black']

    for i in range(6):
        ax.axhline(w_np[i], color=clr[i], ls=':', label=f'Numpy $v_{i+1}$')
        ax.plot(t, eigvec_fe[:, i], color=clr[i], label=f'Euler $v_{i+1}$')

    ax.set_ylabel('Components of vector, $v$')
    ax.set_xlabel('Time, $t$')
    ax.set_title('Prediction of eigenvectors - Forward Euler', fontsize=20)
    ax.legend(loc='lower center', fancybox=True, 
                borderaxespad=0, ncol=3)
    
    fname = r'FE_eigvec_T%i_N%i_dim6' %(args.tot_time, args.N_t_points)
    show_save(fig, fname, args)

    # Plot eigenvalues
    fig, ax = plt.subplots()
    ax.axhline(np.max(v), color='red', ls='--')
    ax.plot(t, eigval_fe)
    ax.plot(s, eigval_nn)

    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Rayleigh Quotient, $r$')
    ax.set_title('Prediction $r_{\\mathrm{final}}$ of \
                largest eigenvalue $\\lambda_{\\mathrm{max}}$', fontsize=20)
    lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
    str(round(np.max(v), 5))
    lgd_euler = "Euler $r_{\\mathrm{final}} \\sim$ " + \
        str(round(eigval_fe[-1], 5))
    lgd_nn = "NN $r_{\\mathrm{final}} \\sim$ " + \
        str(round(eigval_nn.numpy()[-1], 5))
    ax.legend([lgd_numpy, lgd_euler, lgd_nn], loc='lower center',
            fancybox=True, borderaxespad=0, ncol=1)

    fname = r'eigval_T%i_N%i_dim6' %(args.tot_time, args.N_t_points)
    show_save(fig, fname, args)