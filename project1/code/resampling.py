from sklearn.model_selection import train_test_split as tts
# from analysis import MSE
import utils


def NoResampling(X, z, ttsplit, unused_iter_variable, unused_lmd_range, reg_method):
    X_train, X_test, z_train, z_test = tts(X, z, test_size=ttsplit)
    beta = reg_method(X_train, z_train)
    test_prediction = X_test @ beta
    return utils.MSE(z_test, test_prediction)


def Bootstrap(X, z, B, lmd_range, reg_method):
    pass


def cross_validation(X, z, k, lmd_range, reg_method):
    pass
