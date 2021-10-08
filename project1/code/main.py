import utils
import analysis
import argparse

utils.np.random.seed(7132)
valid_funcs = utils.get_directly_implemented_funcs(analysis)

def parameter_range(inp, method, lmb=False):
    """
    For polynomial degree and lambda parameter.
    Returns a string of a single value, a list, or an arange/logspace
    """
    if lmb:
        ntype = float
        func = "utils.np.logspace"
        inp = inp.replace("m", "-")
    else:
        ntype = int
        func = "utils.np.arange"

    if method == "value":
        return f"({ntype(inp)},)"  # tuple with single element
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

    add_arg("-a", "--analyse",
            type=str,
            default="simple_regression",
            help="what analysis function to run",
            )

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
            default="5",
            help='Polynomial degree.',
            )

    add_arg('-pc', '--polynomial-conversion',
            type=str,
            default="range",
            choices=["value", "list", "range"],
            help="How to transform polynomial input",
            )

    add_arg('-n', '--num_points',
            type=int,
            default=30,
            help='Number of gridpoints along 1 axis',
            )

    add_arg('-s', '--scaling',
            type=str,
            default='S',
            choices=["None", 'M', 'S', 'N'],
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
            help="Unused for NoResampling, B for Bootstrap, k.fold for CV",
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
            help="Path to SRTM data file",
            )

    add_arg("-e", "--epsilon",
            type=float,
            default=0.2,
            help="Scale value of noice for Franke Function",
            )

    add_arg("--show",
            dest="show",
            action="store_true",
            )

    add_arg("--noshow",
            action="store_false",
            dest="show",
            )
    parser.set_defaults(show=False)

    args = parser.parse_args(args)

    args.polynomial = parameter_range(args.polynomial, args.polynomial_conversion)
    args.lmb = parameter_range(args.lmb, args.lambda_conversion, True)

    print("Runtime arguments:", args, "\n")

    args.polynomial = eval(args.polynomial)
    args.lmb = eval(args.lmb)

    return args


def main():
    args = parse_args()
    func = args.analyse
    if func in valid_funcs.keys():
        valid_funcs[func](args)
    else:
        print(f"{func} not a valid implemented function")


def terrain():
    """
    DENNE BARE TESTER OG SER PÅ DATAEN GITT I SISTE OPPGAVE
    SLETT SENERE
    """
    from imageio import imread
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    # Load the terrain
    terrain1 = imread('../DataFiles/SRTM_data_Norway_1.tif')
    # Show the terrain
    plt.figure()
    plt.title('Terrain over Norway 1')
    print(type(terrain1))
    print(np.shape(terrain1))
    plt.imshow(terrain1, cmap='gray')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
if __name__ == "__main__":
    main()
    # terrain()
