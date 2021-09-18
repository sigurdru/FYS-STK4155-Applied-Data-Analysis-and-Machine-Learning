import numpy as np
import sys

np.random.seed(136)


def FrankeFunction(x, y, eps0=0):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7)**2) / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-((9 * x - 4)**2) - (9 * y - 7)**2)
    noise = eps0 * np.random.normal(size=x.shape)
    return term1 + term2 + term3 + term4 + noise


def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X


def main():
    N = 100
    P = 5
    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    x, y = np.meshgrid(x, y)

    X = create_X(x, y, P)
    print(X.shape)

if __name__ == "__main__":
    main()
