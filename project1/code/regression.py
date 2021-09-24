import numpy as np
from sklearn import linear_model


def Ordinary_least_squares(X, z, lmd=0):
    return np.linalg.pinv(X.T @ X) @ (X.T @ z)


def Ridge(X, z, lmd):
    return np.linalg.pinv(X.T @ X + lmd * np.eye(X.shape[1])) @ X.T @ z


def Lasso(X, z, lmd):
    model = linear_model.Lasso(lmd)
    model.fit(X, z)
    return model.coef_

