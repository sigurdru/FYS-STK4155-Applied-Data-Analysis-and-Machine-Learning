import utils
from analysis import tmp_func_name


def main():
    args = utils.parse_args()

    tmp_func_name(args)

if __name__ == "__main__":
    main()
