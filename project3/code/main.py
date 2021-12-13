import argparse
import numpy as np
import analysis


def parse_args(args=None):
    """
    Uses argparse module to return an object containing
    all runtime arguments specified in command line
    """
    parser = argparse.ArgumentParser(
        description='Numerical solutions of differential equations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arg = parser.add_argument

    add_arg('-m', '--method',
            type=str,
            default='Euler',
            choices=['Euler', 'NN', 'Eig'],
            help='Choose which prediction method to use.',
            )

    add_arg('-dx', '--x_step',
            type=float,
            default=0.01,
            help='Steplength in x-direction',
            )

    add_arg('-dt', '--t_step',
            type=float,
            default=0,
            help='Steplength in time. Calculated from stability criterion if set to zero',
            )

    add_arg('-sc', '--stability_criterion',
            type=float,
            default=0.5,
            help='Stability criterion of solver. Determines dt if dt=0',
            )

    add_arg('-T', '--tot_time',
            type=float,
            default=1.,
            help='End time, start time = 0',
            )

    add_arg('-BC_l', '--left_boundary_condition',
            type=float,
            default=0,
            help='left boundary condition', 
            )

    add_arg('-BC_r', '--right_boundary_condition',
            type=float,
            default=0,
            help='right boundary condition',
            )
    add_arg('-Np', '--num_plots',
            type=int,
            default=5,
            help='Number of times one wants to plot the evolution',
           )

    add_arg('-N0', '--num_initial_points',
            type=int,
            default=50,
            help='Number of points used for initial values' 
           )

    add_arg('-Nb', '--num_boundary_points',
            type=int,
            default=50,
            help='Number of points used for boundary values'
            )

    add_arg('-Nr', '--num_train_points',
            type=int,
            default=10000,
            help='Number of points used for training each step'
            )

    add_arg('-Nh', '--num_hidden_layers',
            type=int,
            default=8,
            help='Number of hidden layers'
            )

    add_arg('-Nn', '--num_neurons_per_layer',
            type=int,
            default=20,
            help='Number of neurons in hidden layers'
            )

    add_arg('-Nt', '--num_train_iter',
            type=int,
            default=500,
            help='Number of iterations for training'
            )

    add_arg('-dim', '--dimension',
            type=int,
            default=3,
            help='Dimensions of matrix in eigenvalue problem'
            )

    add_arg('-N', '--N_t_points',
            type=int,
            default=101,
            help='Eigenvalue problem: number of timepoints'
            )

    add_arg('-eta', '--learning_rate',
            type=float,
            default=0.005,
            help='Eigenvalue problem: Learning rate for Adam optimizer.'
            )

    add_arg('-epochs', '--nr_epochs',
            type=int,
            default=2000,
            help='Eigenvalue problem: Number of epochs for training neural network.'
            )

    add_arg('-act', '--activation_function',
            type=str,
            default='tanh',
            choices=['tanh'],
            help='Activation function in hidden layers',
            )

    add_arg('-dtype', '--datatype',
            type=str,
            default='float32',
            choices=['float32'],
            help='Activation function in hidden layers',
            )

    add_arg('-study_times',
            action="store_true",
            dest="study_times",
            help='Study two specific times of solver',
           )

    add_arg('-TE', '--test_error',
            action='store_true',
            dest='test_error',
            help='Deviation of numerical solution to analytical solution, tested for two mesh resolutions',
            )

    add_arg("-show",
            action="store_true",
            dest="show",
            )

    add_arg("-push",
            action="store_true",
            dest="push",
            )

    add_arg("-nosave",
            action="store_false",
            dest="save",
            )

    add_arg("-seed",
            type=int,
            default=2021,
            help="Random seed. If 0, no seed is used",
            )
    args = parser.parse_args(args)
    print("Runtime arguments:", args, "\n")
    return args


def main():
    args = parse_args()
    if args.seed:
        np.random.seed(args.seed)

    if args.method == 'Euler':
        u = analysis.forward_euler(args)

    if args.method == 'NN':
        analysis.neural_network(args)
    
    if args.method == 'Eig':
        analysis.neural_network_eig(args)    

if __name__ == "__main__":
    main()
