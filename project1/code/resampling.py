from collections import defaultdict
from typing import DefaultDict
import numpy as np
from sklearn.model_selection import train_test_split as tts, KFold
import utils

def split_scale(X, z, ttsplit, scaler):
    """
    Split and scale data
    z-data is only scaled if scaler is StandardScaler
    Also used to scale data for CV, but this does its own splitting.
    Args:
        X, 2darray: Full design matrix
        z, 2darray: dataset
        ttsplit, float: train/test split ratio
        scaler, sklearn.preprocessing object: Is fitted to train data, scales train and test
    Returns:
        X_train, X_test, z_train, z_test, 2darrays: Scaled train and test data
    """

    if ttsplit != 0:
        X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)
    else:
        X_train = X
        z_train = z
        X_test = X
        z_test = z

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler.fit(z_train)
    z_train = scaler.transform(z_train)
    z_test = scaler.transform(z_test)

    return X_train, X_test, z_train, z_test


def resample(x, z):
    """
    Resamples x and z with replacement
    """
    N = x.shape[0]
    idxs = np.random.randint(0, N, size=(N,))
    x = x[idxs]
    z = z[idxs]
    return x, z


def NoResampling(X, z, ttsplit, unused_iter_variable, lmb, reg_method, scaler, Testing=False):
    X_train, X_test, z_train, z_test = split_scale(X, z, ttsplit, scaler)
    beta = reg_method(X_train, z_train, lmb)
    test_pred = X_test @ beta
    train_pred = X_train @ beta

    data = {}
    data["test_MSE"] = utils.MSE(z_test, test_pred)
    data["train_MSE"] = utils.MSE(z_train, train_pred)
    data["test_R2"] = utils.R2(z_test, test_pred)
    data["train_R2"] = utils.R2(z_train, train_pred)
    if not Testing:
        return data
    else:
        return data, X_train, X_test, z_train, z_test, beta

def Bootstrap(X, z, ttsplit, B, lmb, reg_method, scaler):
    X_train, X_test, z_train, z_test = split_scale(X, z, ttsplit, scaler)

    if B is None:
        B = len(z_train)

    data = DefaultDict(lambda:0)
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

def cross_validation(X, z, unused_tts, k, lmb, reg_method, scaler):
    X, _, z, _ = split_scale(X, z, 0, scaler)  # In this case only scales

    data = defaultdict(lambda: 0)
    train_pred = np.empty(k)
    test_pred = np.empty(k)
    # test_bias = np.empty(k)
    # train_bias = np.empty(k)
    # train_var = np.empty(k)
    # test_var = np.empty(k)

    kfold = KFold(n_splits = k)  # Use sklearns kfold method
    for i, (train_inds, test_inds) in enumerate(kfold.split(X)):
        x_train = X[train_inds]
        z_train = z[train_inds]

        x_test = X[test_inds]
        z_test = z[test_inds]

        beta = reg_method(x_train, z_train, lmb)
        train_p = x_train @ beta
        test_p = x_test @ beta
        train_pred[i] = utils.MSE(z_train, x_train @ beta)
        test_pred[i] = utils.MSE(z_test, x_test @ beta)
        # test_bias[i] = utils.Bias(z_test, test_p)

    data["train_MSE"] = np.mean(train_pred)
    data["test_MSE"] = np.mean(test_pred)
    # data["test_bias"] = np.mean(test_bias)
    return data


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
    z = utils.FrankeFunction(x, y, eps=eps)
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
