import numpy as np

def MSE(target, pred):
    """Returns mean squared error between target and prediction.
    
    Args:
        target (array): target function
        pred (array): predictions
        
    """ 
    return np.mean(np.mean((target - pred)**2, axis=1, keepdims=True))

def bias(target, pred):
    """Returns bias of prediction.
    
    Args:
        target (array): target function
        pred (array): predictions
        
    """ 
    return np.mean((target - np.mean(pred, axis=1, keepdims=True))**2)

def variance(pred):
    """Returns variance of prediction.
    
    Args:
        pred (array): predictions
        
    """ 
    return np.mean((pred - np.mean(pred, axis=1, keepdims=True))**2)

def xyz_1D(x, y, z):
    """Create mesh from inputs, and reshape inputs and outputs to vectors.
    
    Args:
        x (array): one-dimensional x-coordinates
        y (array): one-dimensional y-coordinates
        
    Returns:
        x (array): two-dimensional grid of x-coordinates
        y (array): two-dimensional grid of y-coordinates
        x_1d (array): flattened x-coordinates
        y_1d (array): flattened y-coordinates
        z_1d (array): flattened output vector
    
    """
    x_1d = np.ravel(x).reshape(np.size(x), 1)
    y_1d = np.ravel(y).reshape(np.size(y), 1)

    z_1d = np.ravel(z)
    
    return x_1d, y_1d, z_1d