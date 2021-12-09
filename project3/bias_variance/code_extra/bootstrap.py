from bias_variance_utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.utils import resample

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def FrankeFunction(x,y, noise_std=0.2, add_noise=False):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    np.random.seed(1436) # Use same random number generator
    noise = noise_std*np.random.normal(0, noise_std, (len(x),len(y))) if add_noise else 0
    return term1 + term2 + term3 + term4 + noise


class Bootstrap:

    def __init__(self, x, y, t, args, nr_boot=100, scaler=StandardScaler, scale_target=True):
        """
        x (array): first input dimension
        y (array): second input dimension
        t (array): targets
        """
        self.x_1d, self.y_1d, self.t_1d = xyz_1D(x, y, t) # Flatten targets

        self.X = np.hstack((self.x_1d, self.y_1d)) # stack 1d inputs
        self.scaler = scaler
        self.scale_target = scale_target
        self.nr_boot = nr_boot

        self.X_train, self.X_test, self.t_train, self.t_test \
                    = train_test_split(self.X, self.t_1d, test_size=0.2, random_state=args.seed)

        self.scale(self.scaler, self.scale_target)


    def scale(self, scaler, scale_target):
        """Scale input data (and targets).
        
        Args:
            scaler (func): scaler function from sklearn.preprocessing
            scale_targets (bool): if targets should be scaled or not
            
        """
        scl_X = scaler()
        scl_X.fit(self.X_train)
        self.X_train = scl_X.transform(self.X_train)
        self.X_test = scl_X.transform(self.X_test)
        
        if scale_target:
            scl_t = scaler()
            scl_t.fit(self.t_train.reshape(-1,1))
            self.t_train = scl_t.transform(self.t_train.reshape(-1,1))
            self.t_test = scl_t.transform(self.t_test.reshape(-1,1))
    
    def simulate(self, model):
        """
        Args:
            model: model from scikit library
            B: number of bootstraps
        """
        X_train, X_test, t_train, t_test = self.X_train, self.X_test, self.t_train, self.t_test
        t_tilde = np.zeros((t_train.shape[0], self.nr_boot))
        t_pred = np.zeros((t_test.shape[0], self.nr_boot))

        if len(t_train.shape) > 1:
            t_train = t_train.ravel()
        if len(t_test.shape) > 1:
            t_test = t_test.ravel()
        
        for i in range(self.nr_boot): # Amounts of bootstraps to perform
            X_boot, t_boot = resample(X_train, t_train)
        
            model.fit(X_boot, t_boot)
            t_tilde[:,i] = model.predict(X_train) # reshape to get same dimension as targets
            t_pred[:,i] = model.predict(X_test)

        return t_tilde, t_pred

    def mse_decomposition(self, t_pred):
        mse_test = MSE(self.t_test, t_pred)
        bias_test = bias(self.t_test, t_pred)
        var_test = variance(t_pred)

        return mse_test, bias_test, var_test


