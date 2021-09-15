import numpy as np
from sklearn.model_selection import train_test_split as tts
np.random.seed(136)


def FrankeFunction(x, y, eps0, *args, **kwargs):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = eps0 * np.random.normal(size=x.shape)
    return term1 + term2 + term3 + term4 + noise


def make_design_matrix(x, y, max_pow=3):
    Vandermonde = np.ones((x.shape[0], 1))  # initialize with intercept
    for i in range(1, max_pow + 1):
        for j in range(i + 1):
            Vandermonde = np.c_[Vandermonde, x ** (i - j) * y ** (j)]  # column concatenation
    return Vandermonde[:, 1:]


class Regression:
    def __init__(self, x, f, *args, P=5, train_size=0.8, reg_method="ord_lstsq", resampling=None, scaling=None, **kwargs):
        if "__iter__" in dir(x):
            self.x = x
        else:
            self.x = (x,)

        X = make_design_matrix(*x, P)
        y = f(*x, *args, **kwargs)
        self.X_train, self.X_test, self.y_train, self.y_test = tts(X, y, train_size=train_size)
        print(f"Setting up design matrix with {X.shape[1]} features")

        self.method = f"self.{reg_method}()"

    def regression(self):
        exec(self.method)

    def ord_lstsq(self, *args, **kwargs):
        self.beta = np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
        self.train_prediction = self.X_train @ self.beta
        self.test_prediction = self.X_test @ self.beta

    def MSE(self, test=False):
        if test:
            y = self.y_test
            yp = self.test_prediction
        else:
            y = self.y_train
            yp = self.train_prediction
        return sum((y - yp) ** 2) / len(y)

    def R2(self, test=False):
        if test:
            y = self.y_test
            yp = self.test_prediction
        else:
            y = self.y_train
            yp = self.train_prediction
        return 1 - sum((y - yp) ** 2) / sum((y - np.mean(y)) ** 2)




N = 100
P = 5
x = np.sort(np.random.uniform(size=N))
y = np.sort(np.random.uniform(size=N))

Model = Regression((x, y), FrankeFunction, P=P, eps0=0.1)
Model.regression()

print("Performance of model:")
print(f"MSE train: {Model.MSE()}")
print(f"MSE test: {Model.MSE(True)}")
print(f"R2 train: {Model.R2()}")
print(f"R2 test: {Model.R2(True)}")
