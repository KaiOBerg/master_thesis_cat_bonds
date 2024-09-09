import numpy as np
import pandas as pd


def check_scalar(variable):
    if np.isscalar(variable):
        cor_var = np.array([variable])
    else:
        cor_var = variable
    
    return cor_var

