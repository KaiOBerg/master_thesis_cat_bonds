import pandas as pd
import numpy as np
import re


def init_ws_grid(tc_storms, agg_exp, stat):
    """
    Calculates a specified statistic (mean, max, or median) for each events sustained wind speeds
    from tc_storms for each grid cell.

    Args:
        tc_storms: A hazard object containing wind speed data per centroid.
        agg_exp (dict): A dictionary where keys are the labels of grid cells and values are lists of 
                        line numbers corresponding to indices in tc_storms.intensity.
        stat (str): The statistic to calculate. Can either be a numer to calculate percentile or the
                    string 'mean' to calculate the average.
    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics with labels as columns.
    """
    
    #Initialize a dictionary to hold the calculated statistics
    ws_grid = {letter: [None] * len(tc_storms.event_id) for letter in agg_exp.keys()}
    ws_grid['year'] = [None] * len(tc_storms.event_id)

    #Iterate over each event
    for i in range(len(tc_storms.event_id)):
        year_string = tc_storms.event_name[i]
        ws_grid['year'][i] = extract_year(year_string)
        #For each grid cell, calculate the desired statistic
        for letter, line_numbers in agg_exp.items():
            selected_values = tc_storms.intensity[i, line_numbers]

            # Calculate the statistic based on the user's choice
            if stat == 'mean':
                ws_grid[letter][i] = selected_values.mean()
            elif isinstance(stat, (int, float)):
                dense_array = selected_values.toarray()
                flattened_array = dense_array.flatten()
                ws_grid[letter][i] = np.percentile(flattened_array, stat)
            else:
                raise ValueError("Invalid statistic choice. Choose number for percentile or 'mean'")

    # Convert the dictionary to a DataFrame
    ws_grid = pd.DataFrame.from_dict(ws_grid)
    
    return ws_grid


#determine year for every event and add it to dataframe
#Regular expression to capture the number before .txt and the one after .txt-
pattern = r'(\d+)\.txt-(\d+)'

# Function to extract the numbers
def extract_year(string):
    match = re.search(pattern, string)
    if match:
        before_txt = int(match.group(1))
        after_txt = int(match.group(2))
        year = before_txt * 1000 + after_txt
    return year
