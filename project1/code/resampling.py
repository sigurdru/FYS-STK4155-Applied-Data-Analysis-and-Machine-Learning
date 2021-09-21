from matplotlib.pyplot import sca
from sklearn.model_selection import train_test_split as tts
from collections import defaultdict as ddict

from sklearn.utils import resample
import utils
import numpy as np


def NoResampling(X, z, ttsplit, unused_iter_variable, lmd_range, reg_method, scaler):
    X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)

    # z_test = (z_test - np.mean(z_train)) / np.std(z_train)
    # z_train = (z_train - np.mean(z_train)) / np.std(z_train)


    beta = reg_method(X_train, z_train)
    test_prediction = X_test @ beta
    
    data = ddict(None)
    data["test_MSE"] = utils.MSE(z_test, test_prediction)
    data["train_MSE"] = utils.MSE(z_train, X_train @ beta)
    return data

def Bootstrap(X, z, ttsplit, B, lmd_range, reg_method, scaler):
    X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if B is None:
        B = len(z_train)
    
    data = ddict(None)
    z_pred = np.empty((z_test.shape[0], B))
    for i in range(B):
        x, z = resample(X_train, z_train)
        beta = reg_method(x, z)
        z_pred[:, i] = (X_test @ beta).ravel()

    data["error"] = np.mean( np.mean((z_test - z_pred) ** 2, axis=1, keepdims=True) )
    data["bias"] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True)) ** 2 )
    data["variance"] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    return data

def cross_validation(X, z, ttsplit, k, lmd_range, reg_method, scaler):
    pass
