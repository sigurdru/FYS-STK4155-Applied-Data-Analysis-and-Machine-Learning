import utils
from analysis import bias_var_tradeoff, lambda_BVT, simple_regression

utils.np.random.seed(136)


def main():
    args = utils.parse_args()
    # if args.resampling == "None":
    #     simple_regression(args)
    # else:
    #     bias_var_tradeoff(args)
    lambda_BVT(args)

if __name__ == "__main__":
    main()
