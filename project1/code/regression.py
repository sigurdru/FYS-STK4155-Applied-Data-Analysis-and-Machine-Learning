import numpy as np
from sklearn import linear_model


def Ordinary_least_squares(X, z, lmb=0):
    return np.linalg.pinv(X.T @ X) @ (X.T @ z)


def Ridge(X, z, lmb):
    return np.linalg.pinv(X.T @ X + lmb * np.eye(X.shape[1])) @ X.T @ z


def Lasso(X, z, lmb):
    model = linear_model.Lasso(lmb)
    model.fit(X, z)
    return model.coef_.reshape(-1, 1)

