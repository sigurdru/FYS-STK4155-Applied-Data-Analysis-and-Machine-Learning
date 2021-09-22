from sklearn.model_selection import train_test_split as tts
from collections import defaultdict as ddict

from sklearn.utils import resample
import utils
import numpy as np


def NoResampling(X, z, ttsplit, unused_iter_variable, lmd_range, reg_method, scaler):
    X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)

    # Scaling
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    z_test = (z_test - np.mean(z_train)) / np.std(z_train)
    z_train = (z_train - np.mean(z_train)) / np.std(z_train)

    beta = reg_method(X_train, z_train)
    test_pred = X_test @ beta
    train_pred = X_train @ beta

    data = {}
    data["test_MSE"] = utils.MSE(z_test, test_pred)
    data["train_MSE"] = utils.MSE(z_train, train_pred)

    return data


def Bootstrap(X, z, ttsplit, B, lmd_range, reg_method, scaler):
    X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if B is None:
        B = len(z_train)

    data = {}
    test_pred = np.empty((z_test.shape[0], B))
    train_pred = np.empty((z_train.shape[0], B))
    for i in range(B):
        x, z = resample(X_train, z_train)
        beta = reg_method(x, z)
        test_pred[:, i] = (X_test @ beta).ravel()
        train_pred[:, i] = (X_train @ beta).ravel()

    data["test_MSE"] = utils.MSE_boot(z_test, test_pred)
    data["test_bias"] = utils.Bias(z_test, test_pred)
    data["test_variance"] = utils.Variance(z_test, test_pred)
    data["train_MSE"] = utils.MSE_boot(z_train, train_pred)
    data["train_bias"] = utils.Bias(z_train, train_pred)
    data["train_variance"] = utils.Variance(z_train, train_pred)
    return data

def cross_validation(X, z, ttsplit, k, lmd_range, reg_method, scaler):
    pass
