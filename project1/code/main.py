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
    terrain1 = imread('SRTM_data_Norway_1.tif')
    # Show the terrain
    plt.figure()
    plt.title('Terrain over Norway 1')
    print(type(terrain1))
    print(np.shape(terrain1))
    # plt.imshow(terrain1, cmap='gray')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
if __name__ == "__main__":
    # main()
    terrain()
