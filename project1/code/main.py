# import argparse
# import ord_lstsq, plot
import numpy as np
import utils
from analysis import tmp_func_name


def main():
    args = utils.parse_args()

    tmp_func_name(args)
    # x = np.sort(np.random.uniform(size=n))
    # y = np.sort(np.random.uniform(size=n))
    # Model = ord_lstsq.Regression((x, y), ord_lstsq.FrankeFunction, P=p, eps0=0, scaling="S")
    # Model.fit()

if __name__ == "__main__":
    main()
