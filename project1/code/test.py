import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split as tts


#Our files
import resampling
import analysis
import utils
import regression
def f(x, eps=0):
    """
    Returns the function used for testing of the methods, with noise.

    Args:
        x (array): array of x values
        eps (float): size of error
    """
    value = np.exp(x)
    value += eps*np.random.normal(0,1, size=len(x))
    return value

def create_design_matrix(x,p):
    """
    Here we create a simple design matrix using a polynomial fit, i.e 
    f(x) = a0 + a1*x^2 + a2*x^3 + ... + a(p-1)*x^(p-1)
    
    Args:
        x(array): array of x values
        p(int): max polynomial degree
    """
    X = np.zeros((len(x),p))
    for i in range(p):
        X[:,i] = x**p
    return X
def test_OLS():
    """
    Here we perform a test on our ordinary least square method
    using sklearn
    """
    #polynomial degree
    p_max = 100
    p_array = np.arange(1, p_max, 1)
    our_MSE = np.zeros(p_max-1)
    sklearn_MSE = np.zeros(p_max-1)
    #the noise we want
    eps = 0.2
    #number of data points
    n = 100
    #percentage of data used for testing
    ttsplit = 0.2
    #creating data
    x = np.sort(np.random.uniform(size=n))
    z = f(x, eps)
    for p in p_array:
        X = create_design_matrix(x, p)  
        data, X_train, X_test, z_train, z_test, beta = resampling.NoResampling(X, z, ttsplit, 0, [0],
                                                                           regression.Ordinary_least_squares,
                                                                           scaler=analysis.scale_conv["S"],
                                                                           Testing=True)
        our_MSE[p-1] = data["test_MSE"]

        regOLS = LinearRegression(fit_intercept=True)
        regOLS.fit(X_train, z_train)
        OLS_predict = regOLS.predict(X_test)
        sklearn_MSE[p-1] = utils.MSE(z_test, OLS_predict)
    plt.title('OLS')
    plt.plot(p_array, np.abs(our_MSE - sklearn_MSE), label='Our MSE')
    # plt.plot(p_array, np.log(sklearn_MSE), label='Sklearn MSE')
    plt.legend()
    plt.show()
def test_Ridge():
    """
    Here we perform a test on our Ridge method
    using sklearn
    """
    #polynomial degree
    p_max = 100
    p_array = np.arange(1, p_max, 1)
    our_MSE = np.zeros(p_max-1)
    sklearn_MSE = np.zeros(p_max-1)
    #the noise we want
    eps = 0.2
    #number of data points
    n = 100
    lmd = 0.2
    #percentage of data used for testing
    ttsplit = 0.2
    #creating data
    x = np.sort(np.random.uniform(size=n))
    z = f(x, eps)
    for p in p_array:
        X = create_design_matrix(x, p)
        data, X_train, X_test, z_train, z_test, beta = resampling.NoResampling(X, z, ttsplit, 0, [lmd],
                                                                               regression.Ridge,
                                                                               scaler=analysis.scale_conv["S"],
                                                                               Testing=True)
        our_MSE[p-1] = data["test_MSE"]

        regOLS = Ridge(fit_intercept=True)
        regOLS.fit(X_train, z_train)
        OLS_predict = regOLS.predict(X_test)
        sklearn_MSE[p-1] = utils.MSE(z_test, OLS_predict)
    plt.title('Ridge')
    plt.plot(p_array, np.log(our_MSE), label='Our MSE')
    plt.plot(p_array, np.log(sklearn_MSE), label='Sklearn MSE')
    plt.legend()
    plt.show()
def test_Lasso():
    """
    Here we perform a test on our Lasso method
    using sklearn
    """
    #polynomial degree
    p_max = 100
    p_array = np.arange(1, p_max, 1)
    our_MSE = np.zeros(p_max-1)
    sklearn_MSE = np.zeros(p_max-1)
    #the noise we want
    eps = 0.2
    #number of data points
    n = 100
    lmd = 0.2
    #percentage of data used for testing
    ttsplit = 0.2
    #creating data
    x = np.sort(np.random.uniform(size=n))
    z = f(x, eps)
    for p in p_array:
        X = create_design_matrix(x, p)
        data, X_train, X_test, z_train, z_test, beta = resampling.NoResampling(X, z, ttsplit, 0, [lmd],
                                                                               regression.Lasso,
                                                                               scaler=analysis.scale_conv["S"],
                                                                               Testing=True)
        our_MSE[p-1] = data["test_MSE"]

        regOLS = Lasso(fit_intercept=True)
        regOLS.fit(X_train, z_train)
        OLS_predict = regOLS.predict(X_test)
        sklearn_MSE[p-1] = utils.MSE(z_test, OLS_predict)
    plt.title('Lasso')
    plt.plot(p_array, np.log(our_MSE), label='Our MSE')
    plt.plot(p_array, np.log(sklearn_MSE), label='Sklearn MSE')
    plt.legend()
    plt.show()

class params:
    """
    This class is so we can create a series of values our functions use
    """
    def __init__(self, N, P, eps, resampling,
                 tts, resampling_iter, lmb, method,
                 scaling_conv):
        self.num_points = N
        self.polynomial = P
        self.epsilon = eps
        self.resampling = resampling
        self.tts = tts
        self.resampling_iter = resampling_iter
        self.lmb = lmb
        self.method = method
        self.scaling = scaling_conv
        self.dataset = "Test"
def test_CV():
    """
    Test cross va
    """
    N = 50
    P = np.arange(1,5,1)
    eps = 0.2
    resampling = "Boot"
    tts = 0.2
    resampling_iter = 10
    lmb = [0.1]
    method = "OLS"
    scaling_conv = "S"
    args = params(N, P, eps, 
                resampling, tts, resampling_iter, 
                lmb, method, scaling_conv)
    our_results = analysis.bias_var_tradeoff(args, testing=True)
    plt.plot(P, our_results["test_biases"], label = "test b")
    plt.plot(P, our_results["test_vars"], label = "test v")
    plt.plot(P, our_results["train_biases"], label = "train b")
    plt.plot(P, our_results["train_vars"], label = "train v")
    plt.legend()
    plt.show()

def test_bootstrap():
    """
    test bootstrap
    """
    pass
def plot_test_func():
    """
    Here we plot the test data along with the analytical test function
    """
    n = 100
    eps = 0.2
    x = np.sort(np.random.uniform(size=n))
    plt.plot(x, f(x,eps), label='with noise')
    plt.plot(x, f(x), label='no noise')
    plt.legend()
    plt.show()

plot_test_func()
test_OLS()
test_Ridge()
test_Lasso()
test_CV()

