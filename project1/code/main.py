import utils
import analysis

utils.np.random.seed(136)
valid_funcs = utils.get_directly_implemented_funcs(analysis)

def main():
    args = utils.parse_args()
    func = args.analyse
    if func in valid_funcs.keys():
        valid_funcs[func](args)
    else:
        print(f"{func} not a valid implemented function")

if __name__ == "__main__":
    main()
