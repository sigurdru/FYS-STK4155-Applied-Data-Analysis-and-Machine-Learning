from collections import defaultdict
import numpy as np
from sklearn.model_selection import KFold
import utils

# Different scoring functions
def MSE(y, y_pred):
    return sum((y - y_pred) ** 2) / len(y)

def MSE_boot(y, y_pred):
    return np.mean( np.mean((y - y_pred) ** 2, axis=1, keepdims=True) )

def R2(y, y_pred):
    return 1 - sum((y - y_pred) ** 2) / sum((y - np.mean(y)) ** 2)

def Bias(y, y_pred):
    return np.mean( (y - np.mean(y_pred, axis=1, keepdims=True)) ** 2 )

def Variance(y, y_pred):
    return np.mean( np.var(y_pred, axis=1, keepdims=True) )

def resample(x, z):
    """
    Resamples x and z with replacement
    """
    N = x.shape[0]
    idxs = np.random.randint(0, N, size=(N,))
    x = x[idxs]
    z = z[idxs]
    return x, z


def NoResampling(X, z, unused_iter_variable, lmb, reg_method, Testing=False):
    """
    Performs regression without resampling. 
    X and z should be pre-split and scaled.

    Args:
        X, 2tuple: 
            contains X_train and X_test.
        z, 2tuple:
            contains z_train and z_test
        unused_iter_variable, any:
            Taken for compatibility with Bootstrapping and CV. Unused
        lmb, float:
            lambda-parameter for Ridge and Lasso regression
        reg_method, callable:
            Function object from Regression. Is one of 3 regression methods
        Testing, bool:
            If True, return more data for testing with SKlearn reasons
    Returns:
        data, dict:
            Dictionary containing train and test MSE and R2, as well at the beta parameters.
            If testing, also return train and test X and z. 
    """
    X_train, X_test = X
    z_train, z_test = z
    beta = reg_method(X_train, z_train, lmb)
    test_pred = X_test @ beta
    train_pred = X_train @ beta

    data = defaultdict(lambda: 0)
    data["beta"] = beta
    data["test_MSE"] = MSE(z_test, test_pred)
    data["train_MSE"] = MSE(z_train, train_pred)
    data["test_R2"] = R2(z_test, test_pred)
    data["train_R2"] = R2(z_train, train_pred)
    if not Testing:
        return data
    else:
        return data, X_train, X_test, z_train, z_test, beta

def Bootstrap(X, z, B, lmb, reg_method):
    """
    Performs regression with bootstrapping. 
    X and z should be pre-split and scaled

    Args:
        X, 2tuple: 
            contains X_train and X_test.
        z, 2tuple:
            contains z_train and z_test
        B, int:
            Number of bootstrapping iterations
        lmb, float:
            lambda-parameter for Ridge and Lasso regression
        reg_method, callable:
            Function object from Regression. Is one of 3 regression methods
    Returns:
        data, dict:
            Dictionary containing train and test MSE, bias and variance.
    """
    X_train, X_test = X
    z_train, z_test = z

    if B is None:
        B = len(z_train)

    data = defaultdict(lambda:0)
    test_pred = np.empty((z_test.shape[0], B))
    train_pred = np.empty((z_train.shape[0], B))

    for i in range(B):
        x, z = resample(X_train, z_train)
        beta = reg_method(x, z, lmb)
        test_pred[:, i] = (X_test @ beta).ravel()
        train_pred[:, i] = (X_train @ beta).ravel()

    data["test_MSE"] = MSE_boot(z_test, test_pred)
    data["test_bias"] = Bias(z_test, test_pred)
    data["test_variance"] = Variance(z_test, test_pred)

    data["train_MSE"] = MSE_boot(z_train, train_pred)
    data["train_bias"] = Bias(z_train, train_pred)
    data["train_variance"] = Variance(z_train, train_pred)
    return data

def cross_validation(X, z, k, lmb, reg_method):
    """
    Performs regression with k-fold cross-validation. 
    X and z should be pre-scaled, but not split

    Args:
        X, 2darray: 
            contains full design matrix.
        z, 2darray:
            contains full data
        k, int:
            Number of CV iterations
        lmb, float:
            lambda-parameter for Ridge and Lasso regression
        reg_method, callable:
            Function object from Regression. Is one of 3 regression methods
    Returns:
        data, dict:
            Dictionary containing train and test MSE.
    """
    data = defaultdict(lambda: 0)
    train_pred = np.empty(k)
    test_pred = np.empty(k)

    kfold = KFold(n_splits = k)  # Use sklearns kfold method
    for i, (train_inds, test_inds) in enumerate(kfold.split(X)):
        x_train = X[train_inds]
        z_train = z[train_inds]

        x_test = X[test_inds]
        z_test = z[test_inds]

        beta = reg_method(x_train, z_train, lmb)
        train_pred[i] = MSE(z_train, x_train @ beta)
        test_pred[i] = MSE(z_test, x_test @ beta)

    data["train_MSE"] = np.mean(train_pred)
    data["test_MSE"] = np.mean(test_pred)
    return data


if __name__=='__main__':
    """
    Here we test the different methods! :)
    Does this work? Think not properly.
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

    all_data = tts(X, z, test_size=ttsplit)
    X_train, X_test, z_train, z_test = all_data
    #Testing
    data = NoResampling(all_data, 0, 0, regression.Ordinary_least_squares)
    MSE_test = data["test_MSE"]

    regOLS = LinearRegression(fit_intercept=False)
    regOLS.fit(X_train, z_train)
    OLS_predict = regOLS.predict(X_test)
    MSE_test_sk = MSE(z_test, OLS_predict)
    print('Test error')
    print(MSE_test, MSE_test_sk)
