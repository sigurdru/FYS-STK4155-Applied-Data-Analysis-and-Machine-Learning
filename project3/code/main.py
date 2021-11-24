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
            choices=['Euler', 'NN'],
            help='Choose which prediction method to use.',
            )

    add_arg('-dx', '--num_x_points',
            type=float,
            default=0.005,
            help='Steplength in x-direction',
            )

    add_arg('-dt', '--num_t_points',
            type=float,
            default=0.0001,
            help='Steplength in time',
            )

    add_arg('-T', '--tot_time',
            type=float,
            default=1,
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
        analysis.forward_euler(args)

    if args.method == 'NN':
        analysis.neural_network(args)

if __name__ == "__main__":
    main()