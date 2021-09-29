import utils
from analysis import bias_var_tradeoff, no_resamp

utils.np.random.seed(136)


def main():
    args = utils.parse_args()
    if args.resampling == "None":
        no_resamp(args)
    else:
        bias_var_tradeoff(args)

if __name__ == "__main__":
    main()
