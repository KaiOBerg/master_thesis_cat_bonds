import pandas as pd
import numpy as np

def init_ws_grid(tc_storms, agg_exp, stat='max'):
    """
    Calculates a specified statistic (mean, max, or median) for each events sustained wind speeds
    from tc_storms for each grid cell.

    Args:
        tc_storms: A hazard object containing wind speed data per centroid.
        agg_exp (dict): A dictionary where keys are the labels of grid cells and values are lists of 
                        line numbers corresponding to indices in tc_storms.intensity.
        stat (str): The statistic to calculate. Options are 'mean', 'max', or 'median'.
                    Default is 'max'.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics with labels as columns.
    """
    
    #Initialize a dictionary to hold the calculated statistics
    ws_grid = {letter: [None] * len(tc_storms.event_id) for letter in agg_exp.keys()}

    #Iterate over each event
    for i in range(len(tc_storms.event_id)):
        #For each grid cell, calculate the desired statistic
        for letter, line_numbers in agg_exp.items():
            selected_values = tc_storms.intensity[i, line_numbers]

            # Calculate the statistic based on the user's choice
            if stat == 'mean':
                ws_grid[letter][i] = selected_values.mean()
            elif stat == 'median':
                ws_grid[letter][i] = np.median(selected_values)
            elif stat == 'max':
                ws_grid[letter][i] = selected_values.max()
            else:
                raise ValueError("Invalid statistic choice. Choose 'mean', 'max', or 'median'.")

    # Convert the dictionary to a DataFrame
    ws_grid = pd.DataFrame.from_dict(ws_grid)
    
    return ws_grid
