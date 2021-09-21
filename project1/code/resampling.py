from sklearn.model_selection import train_test_split as tts
from collections import defaultdict as ddict
import utils
import numpy as np


def NoResampling(X, z, ttsplit, unused_iter_variable, lmd_range, reg_method, scaler):
    X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    beta = reg_method(X_train, z_train)
    test_prediction = X_test @ beta
    
    data = ddict(None)
    data["test_MSE"] = utils.MSE(z_test, test_prediction)
    data["train_MSE"] = utils.MSE(z_train, X_train @ beta)
    return data

def Bootstrap(X, z, B, lmd_range, reg_method):
    pass


def cross_validation(X, z, k, lmd_range, reg_method):
    pass
