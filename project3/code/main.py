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

    add_arg('-study_times',
            action="store_true",
            dest="study_times",
            help='Study two specific times of solver',
           )

    add_arg('-TE', '--test_error',
            type=bool,
            default=False,
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
        u, error = analysis.forward_euler(args)

    if args.test_error:
        args.x_step = 0.1 # coarse mesh
        args.t_step = 0.3*args.x_step**2 # ensure stability
        u_coarse, error_coarse = analysis.forward_euler(args)
        print(f'Numerical error for dx={args.x_step}, accumulated for n={args.num_plots} time steps:', error_coarse)

        args.x_step = 0.01 # fine mesh
        args.t_step = 0.3*args.x_step**2 # ensure stability
        u_fine, error_fine = analysis.forward_euler(args)
        print(f'Numerical error for dx={args.x_step}, accumulated for n={args.num_plots} time steps:', error_fine)

    if args.method == 'NN':
        analysis.neural_network(args)

if __name__ == "__main__":
    main()