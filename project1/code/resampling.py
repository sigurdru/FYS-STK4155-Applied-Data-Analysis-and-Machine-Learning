from sklearn.model_selection import train_test_split as tts
from collections import defaultdict as ddict

from sklearn.utils import resample
import utils
import numpy as np


def split_scale(X, z, ttsplit, scaler):
    """
    Split and scale data
    """
    X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    z_test = (z_test - np.mean(z_train)) / np.std(z_train)
    z_train = (z_train - np.mean(z_train)) / np.std(z_train)

    return X_train, X_test, z_train, z_test


def NoResampling(X, z, ttsplit, unused_iter_variable, lmb, reg_method, scaler):
    X_train, X_test, z_train, z_test = split_scale(X, z, ttsplit, scaler)

    beta = reg_method(X_train, z_train, lmd_range[0])
    test_pred = X_test @ beta
    train_pred = X_train @ beta

    data = {}
    data["test_MSE"] = utils.MSE(z_test, test_pred)
    data["train_MSE"] = utils.MSE(z_train, train_pred)
    data["test_R2"] = utils.R2(z_test, test_pred)
    data["train_R2"] = utils.R2(z_train, train_pred)

    return data


def Bootstrap(X, z, ttsplit, B, lmb, reg_method, scaler):
    X_train, X_test, z_train, z_test = split_scale(X, z, ttsplit, scaler)

    if B is None:
        B = len(z_train)

    data = {}
    test_pred = np.empty((z_test.shape[0], B))
    train_pred = np.empty((z_train.shape[0], B))
    for i in range(B):
        x, z = resample(X_train, z_train)
        beta = reg_method(x, z, lmb)
        test_pred[:, i] = (X_test @ beta).ravel()
        train_pred[:, i] = (X_train @ beta).ravel()

    data["test_MSE"] = utils.MSE_boot(z_test, test_pred)
    data["test_bias"] = utils.Bias(z_test, test_pred)
    data["test_variance"] = utils.Variance(z_test, test_pred)
    data["train_MSE"] = utils.MSE_boot(z_train, train_pred)
    data["train_bias"] = utils.Bias(z_train, train_pred)
    data["train_variance"] = utils.Variance(z_train, train_pred)
    return data

def cross_validation(X, z, ttsplit, k, lmb, reg_method, scaler):
    pass


if __name__=='__main__':
    """
    Here we test the different methods! :)
    """
    from sklearn.linear_model import LinearRegression
    import regression
    #Setup
    n = 10
    p = 9
    eps = 1
    ttsplit = 0.2
    x = np.sort(np.random.uniform(size=n))
    y = np.sort(np.random.uniform(size=n))
    x, y = np.meshgrid(x, y)
    z = utils.FrankeFunction(x, y, eps0=eps)
    X = utils.create_X(x, y, p)

    all_data= tts(X, z, test_size=ttsplit)
    X_train, X_test, z_train, z_test = all_data
    #Testing
    data = NoResampling(all_data, 0, 0, regression.Ordinary_least_squares)
    MSE_test = data["test_MSE"]

    regOLS = LinearRegression(fit_intercept=False)
    regOLS.fit(X_train, z_train)
    OLS_predict = regOLS.predict(X_test)
    MSE_test_sk = utils.MSE(z_test, OLS_predict)
    print('Test error')
    print(MSE_test, MSE_test_sk)


