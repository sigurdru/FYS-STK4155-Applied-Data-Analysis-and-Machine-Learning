import numpy as np
from sklearn import linear_model


def Ordinary_least_squares(X, z, lmb=0):
    """
    Performs OLS regression

    Args:
        X, 2darray: design matrix
        z, 2darray: datapoints
        lmb, any: taken for compatibility reasons. Unused
    Returns:
        beta, 1darray; optimal estimators
    """
    return np.linalg.pinv(X.T @ X) @ (X.T @ z)


def Ridge(X, z, lmb):
    """
    Performs Ridge regression

    Args:
        X, 2darray: design matrix
        z, 2darray: datapoints
        lmb, float: lambda-parameter
    Returns:
        beta, 1darray; optimal estimators
    """
    return np.linalg.pinv(X.T @ X + lmb * np.eye(X.shape[1])) @ X.T @ z


def Lasso(X, z, lmb):
    """
    Performs Lasso regression using SKlearn lineaR-model.Lasso 

    Args:
        X, 2darray: design matrix
        z, 2darray: datapoints
        lmb, any: lambda-parameters
    Returns:
        beta, 1darray; optimal estimators
    """
    model = linear_model.Lasso(lmb, max_iter=1e6, tol=1e-1)
    model.fit(X, z)
    return model.coef_.reshape(-1, 1)
