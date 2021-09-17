import numpy as np
import sys


def FrankeFunction(x, y, eps0=0):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7)**2) / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-((9 * x - 4)**2) - (9 * y - 7)**2)
    noise = eps0 * np.random.normal(size=x.shape)
    return term1 + term2 + term3 + term4 + noise


def make_design_matrix(x, y, max_pow=3, intercept=False):
    Vandermonde = np.ones((x.shape[0], 1))  # initialize with intercept
    for i in range(1, max_pow + 1):
        for j in range(i + 1):
            Vandermonde = np.c_[Vandermonde,
                                x**(i - j) * y**(j)]  # column concatenation
    return Vandermonde[:, int(not intercept):]


def main():
    N = 100
    P = 5
    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))

    X = make_design_matrix(x, y, P)
    print(X.shape)


if __name__ == "__main__":
    main()
