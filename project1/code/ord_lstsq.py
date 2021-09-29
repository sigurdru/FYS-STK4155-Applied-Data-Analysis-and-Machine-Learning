import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler as MScaler, StandardScaler as SScaler, Normalizer as NScaler
import matplotlib.pyplot as plt
import sys

# np.random.seed(136)


def FrankeFunction(x, y, eps0=None):
    """
    Returns Franke Function with or without noise

    Args:
        x (array): Array with x-values
        y (array): Array with y-values
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = eps0 * np.random.normal(size=x.shape)
    return term1 + term2 + term3 + term4 + noise


def create_X(x, y, n):
    """
    Sets up design matrix

    Parameters:
        x, y: array-like
            Are flattened if not already
        n: int
            max polynomial degree
    Returns:
        X: 2darray
            Includes intercept.
    """
    if not 1 in x.shape:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = (n + 1) * (n + 2) // 2  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = i * (i + 1) // 2
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)
    return X


# class Regression:
#     def __init__(self, x, f, *args, P=5, train_size=0.8, reg_method="ord_lstsq", resampling=None, scaling=None, **kwargs):
#         self.x = x

#         X = make_design_matrix(*x, P)
#         y = f(*x, *args, **kwargs)
#         self.y = y
#         self.f = f


#         print(f"Setting up design matrix with {X.shape[1]} features")
#         self.X_train, self.X_test, self.y_train, self.y_test = tts(X, y, train_size=train_size)
#         if scaling:
#             scaler = eval(f"{scaling}Scaler()")
#             scaler.fit(self.X_train)
#             self.X_train = scaler.transform(self.X_train)
#             self.X_test = scaler.transform(self.X_test)
#             self.y_train = scaler.transform(self.y_train)
#             self.y_test = scaler.transform(self.y_test)

#         self.fitted = False
#         self.method = eval(f"self.{reg_method}")

#     def fit(self, *args, **kwargs):
#         if self.fitted:
#             print("Model is already fitted!")
#             return
#         else:
#             self.fitted = True
#             self.method(*args, **kwargs)

#     def ord_lstsq(self):
#         self.beta = np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
#         self.train_prediction = self.X_train @ self.beta
#         self.test_prediction = self.X_test @ self.beta

def MSE(y, yp):
    return np.mean( np.mean((y - yp) ** 2, axis=1, keepdims=True) )
    # return np.mean((y - yp) ** 2, axis=1)
    # return np.sum(np.mean(y - yp, axis=1) ** 2, axis=1) / y.shape[1]

def R2(y, yp):
    return 1 - sum((y - yp) ** 2) / sum((y - np.mean(y)) ** 2)


if __name__=='__main__':
    """
    Testing
    """
    N = 100
    P = 20
    P = np.arange(1, P, 2)
    eps = 0.2
    x = np.sort(np.random.uniform(size=N))
    y = np.sort(np.random.uniform(size=N))
    z = FrankeFunction(x, y, eps).flatten()

    B = 100


    scaler = SScaler()
    train_errors = []
    test_errors = []

    for i, p in enumerate(P):
        print(p)
        X = create_X(x, y, p)
        X_train, X_test, z_train, z_test = tts(X, z, test_size=0.2)

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        z_test = (z_test - np.mean(z_train)) / np.std(z_train)
        z_train = (z_train - np.mean(z_train)) / np.std(z_train)

        test_pred = np.zeros((B, len(z_test)))
        train_pred = np.zeros((B, len(z_train)))

        test_err_boot = []

        for i in range(B):
            x_, y_ = resample(X_train, z_train)
            beta = np.linalg.pinv(x_.T @ x_) @ x_.T @ y_

            test_pred[i, :] = X_test @ beta
            train_pred[i, :] = x_ @ beta

        test_errors.append(MSE(z_test, test_pred))
        train_errors.append(MSE(z_train, train_pred))




    plt.plot(P, train_errors, label="Train error")
    plt.plot(P, test_errors, label="test error")
    plt.legend()
    plt.show()



    # print("Performance of model:")
    # print(f"MSE train: {Model.MSE()}")
    # print(f"MSE test: {Model.MSE(True)}")
    # print(f"R2 train: {Model.R2()}")
    # print(f"R2 test: {Model.R2(True)}")
