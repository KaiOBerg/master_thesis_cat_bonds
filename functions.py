import numpy as np
import pandas as pd


def check_scalar(variable):
    if np.isscalar(variable):
        cor_var = np.array([variable])
    else:
        cor_var = variable
    
    return cor_var

def get_all_values(d):
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.extend(get_all_values(value))
        else:
            values.append(value)
    return values

import time
import sys

def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    :param iteration: Current iteration (int)
    :param total: Total iterations (int)
    :param prefix: Prefix string (str)
    :param suffix: Suffix string (str)
    :param decimals: Positive number of decimals in percent complete (int)
    :param length: Character length of bar (int)
    :param fill: Bar fill character (str)
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
    # Print New Line on Complete
    if iteration == total: 
        print()



