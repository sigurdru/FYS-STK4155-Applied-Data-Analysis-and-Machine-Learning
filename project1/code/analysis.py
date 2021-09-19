import numpy as np
import matplotlib.pyplot as plt
import utils
import resampling
import regression


def tmp_func_name(args):
    N = args.num_points
    P = args.polynomial  # polynomial degrees

    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, eps0=args.epsilon).flatten()
    MSEs = np.zeros(len(P))

    for i, p in enumerate(P):
        print("p =", p)
        X = utils.create_X(x, y, p)

        reg_meth = eval(f"regression.{args.method}")
        if args.resampling != "None":
            print("No resampling methods have been implemented")

        MSEs[i] = resampling.NoResampling(X, z, args.tts, args.resampling_iter, args.lmb, reg_meth)

    plt.plot(P, MSEs)
    plt.show()
