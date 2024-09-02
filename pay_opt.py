import pandas as pd
import numpy as np


def slope(x, y, min_pay, max_pay):
    m = (max_pay - min_pay) / (y - x)
    return m

def intercept(nominal, m, y):
    b = nominal - (m * y)
    return b

def calculate_payout(x, y, haz_int, max_min_pay, grid):
    payout_df = pd.DataFrame(index=np.arange(len(haz_int)), columns=np.arange(1))
    for i in range(len(haz_int)):
        filtered_rows = pd.DataFrame(max_min_pay.iloc[grid]) # Filter rows where 'Grid' has the value 'grid'
        max_pay = filtered_rows.loc['Upper'].iloc[0]
        min_pay = filtered_rows.loc['Lower'].iloc[0]
        m = slope(x, y, min_pay, max_pay)
        b = intercept(max_pay, m, y)
        wind_speed = haz_int.iloc[i, grid]

        if x <= wind_speed <= y:
            payout = m * wind_speed + b
        elif wind_speed > y:
            payout = max_pay
        else:
            payout = 0

        payout_df.iloc[i,0] = payout

    return payout_df

def objective_function(params, damage_matrix):
    """Objective function to minimize.
    
    Parameters
    ----------
    params : list
        List of parameters [x, y] for each grid cell.
    damage_matrix : np.ndarray
        Matrix of damage values.
        
    Returns
    -------
    float
        Sum of squared differences between calculated payout and damage.
    """
    num_cells = damage_matrix.shape[1]  # Number of grid cells
    total_diff = 0

    # Extract parameters for each grid cell
    for cell_idx in range(num_cells):
        x = params[cell_idx * 2]      # x for this grid cell
        y = params[cell_idx * 2 + 1]  # y for this grid cell
        
        # Compute payouts
        payouts = calculate_payout(x, y, ws_grid, rp_dam_grid, cell_idx)
        
        # Compute the squared difference
        diff = np.sum(((payouts.iloc[:, 0] - damage_matrix.iloc[:, cell_idx]) ** 2)** 0.5)
        total_diff += diff

    return total_diff