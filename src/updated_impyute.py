# impyute_custom.py

import numpy as np
from functools import wraps

# Error definitions from error.py
class BadInputError(Exception):
    """Error thrown when input args don't match spec"""
    pass

class BadOutputError(Exception):
    """Error thrown when outputs don't match spec"""
    pass

# matrix.py contents
def nan_indices(data):
    """Finds the indices of all missing values."""
    return np.argwhere(np.isnan(data))

# util.py contents with updated _dtype_float function
def _shape_2d(data):
    """True if array is 2D"""
    return len(np.shape(data)) == 2

def _is_ndarray(data):
    """True if the array is an instance of numpy's ndarray"""
    return isinstance(data, np.ndarray)

def _dtype_float(data):
    """True if the values in the array are floating point"""
    return data.dtype == float  # Updated line

def _nan_exists(data):
    """True if there is at least one np.nan in the array"""
    return len(nan_indices(data)) > 0

def constantly(x):
    def func(*args, **kwargs):
        return x
    return func

def complement(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return not fn(*args, **kwargs)
    return wrapper

def execute_fn_with_args_and_or_kwargs(fn, args, kwargs):
    try:
        return fn(*args, **kwargs)
    except TypeError:
        return fn(*args)

# wrapper.py contents
def get_pandas_df():
    try:
        import pandas as pd
        df = pd.DataFrame
    except (ModuleNotFoundError, ImportError):
        df = None
    return df

# Additional decorators from wrapper.py...
# (Add all other decorators and functions from wrapper.py here)

# em.py contents
def em(data, eps=0.1):
    nan_xy = nan_indices(data)
    for x_i, y_i in nan_xy:
        col = data[:, int(y_i)]
        mu = col[~np.isnan(col)].mean()
        std = col[~np.isnan(col)].std()
        col[x_i] = np.random.normal(loc=mu, scale=std)
        previous, i = 1, 1
        while True:
            i += 1
            mu = col[~np.isnan(col)].mean()
            std = col[~np.isnan(col)].std()
            col[x_i] = np.random.normal(loc=mu, scale=std)
            delta = np.abs(col[x_i]-previous)/previous
            if i > 5 and delta < eps:
                data[x_i][y_i] = col[x_i]
                break
            data[x_i][y_i] = col[x_i]
            previous = col[x_i]
    return data

# Add any other functions or content from the other files here
