import argparse
from analysis import analyse
import numpy as np

np.random.seed(2021)

def parse_args(args=None):
    """
    Uses argparse module to return an object containing
    all runtime arguments specified in command line
    """
    parser = argparse.ArgumentParser(
        description='Kort beskrivelse av hva vi gjør dette prosjektet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arg = parser.add_argument

    add_arg('-m', '--method',
            type=str,
            default='OLS',
            choices=['OLS', 'Ridge'],
            help='Choose which regression method to use.',
            )

    add_arg("-tts",
            type=float,
            default=0.2,
            choices=[0.2, 0.25, 0.3, 0.4],
            help="Train/test split ratio"
            )

    add_arg('-p', '--polynomial',
            type=int,
            default=5,
            help='Polynomial degree.',
            )

    add_arg('-n', '--num_points',
            type=int,
            default=30,
            help='Number of gridpoints along 1 axis',
            )

    add_arg('-Ne', '--num_epochs',
            type=int,
            default=100,
            help='Number of epochs',
            )

    add_arg('-eta',
            type=str,
            default='np.logspace(-5,0,6)',
            help="""Desired learning rate, can be array or float.
            For example:
                    -eta 'np.linspace(0.001, 1, 100)'
                    -eta 'np.logspace(0.001, 1, 10)'
                    -eta 1
                    """,
            )

    add_arg('-ga', '--gamma',
            type=float,
            help='Desired momentum parameter'
            )

    add_arg('-ls', '--learning_schedule',
            type=bool,
            default=False,
            help='True if one wants to scale the learning as a function of epochs')

    add_arg('-bs', '--batch_size',
            type=int,
            default=None,
            help='Set size of minibatch'
            )

    add_arg('-s', '--scaling',
            type=str,
            default='S',
            choices=["None", 'M', 'S', 'N'],
            help='Scaling method: None, MinMax, Standard, Normalizer.',
            )

    add_arg("-e", "--epsilon",
            type=float,
            default=0.2,
            help="Scale value of noice for Franke Function",
            )

    add_arg('-l', '--lmb',
            type=str,
            default="0",
            help='Lambda parameter',
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

    add_arg("--show",
            dest="show",
            action="store_true",
            )

    add_arg("--noshow",
            action="store_false",
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

    parser.set_defaults(show=False)

    args = parser.parse_args(args)

    print("Runtime arguments:", args, "\n")

    # for dynamic etas
    exec('args.eta = ' + args.eta)
    if np.shape(args.eta) == ():
        args.eta = [args.eta, ]

    return args


def main():
    args = parse_args()
    analyse(args)


if __name__ == "__main__":
    main()
