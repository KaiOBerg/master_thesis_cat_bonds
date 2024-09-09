import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

sequence = [30, 41]
length = 20
initial_params = np.tile(sequence, length // len(sequence))

#Define bounds for minimum and maximum wind speeds
bounds = [(30, 32) if i % 2 == 0 else (41, 43) for i in range(len(initial_params))]

def slope(x, y, min_pay, nominal):
    m = (nominal - min_pay) / (y - x)
    return m

def intercept(nominal, m, y):
    b = nominal - (m * y)
    return b

def init_optimization(imp_grid_evt_flt, haz_int, max_min_pay, nominal):

    haz_int = haz_int
    max_min_pay = max_min_pay
    nominal = nominal

    #Perform optimization
    result = minimize(
        fun=objective_function,
        x0=initial_params,
        args=(imp_grid_evt_flt, haz_int, max_min_pay, nominal), 
        method='L-BFGS-B',  #How to choose optimization method?
        bounds=bounds  
    )

    #Extract the optimized parameters
    optimized_params = result.x
    print("Optimized parameters:", optimized_params)

    print('Missmatch between payment and damage:',result.fun)

    #Reshape parameters into a more interpretable form if needed
    optimized_xs = optimized_params[0::2]  #minimum threshold of windspeeds
    optimized_ys = optimized_params[1::2]  #maximum threshold of windspeeds

    return result, optimized_xs, optimized_ys

def objective_function(params, damage_matrix, haz_int, max_min_pay, nominal):
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
    num_cells = damage_matrix.shape[1]  #Number of grid cells
    total_diff = 0

    #Extract parameters for each grid cell
    for cell_idx in range(num_cells):
        x = params[cell_idx * 2]      #minimum wind speed
        y = params[cell_idx * 2 + 1]  #maximum wind speed
        
        #Compute payouts
        payouts = calculate_payout(x, y, haz_int, max_min_pay, nominal, cell_idx)
        
        #Compute the difference between payout and damage
        diff = np.sum(((payouts.iloc[:, 0] - damage_matrix.iloc[:, cell_idx]) ** 2)** 0.5)
        total_diff += diff

    return total_diff

def calculate_payout(x, y, haz_int, max_min_pay, nominal, grid):

    payout_df = pd.DataFrame(index=np.arange(len(haz_int)), columns=np.arange(1))
    for i in range(len(haz_int)):
        filtered_rows = pd.DataFrame(max_min_pay.iloc[grid]) #Filter rows where 'Grid' has the value 'grid'
        max_pay = nominal
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

def pay_vs_damage(imp_grid_evt_flt, optimized_xs, optimized_ys, ws_grid, rp_dam_grid, nominal, include_plot=False):
    b = len(imp_grid_evt_flt)
    payout_evt_grd = pd.DataFrame({letter: [None] * b for letter in ws_grid.columns})
    pay_dam_df = pd.DataFrame({'pay': [0] * b, 'damage': [0] * b})

    for i in range(len(imp_grid_evt_flt)):
        tot_dam = np.sum(imp_grid_evt_flt.iloc[i, :])
        pay_dam_df.loc[i,"damage"] = tot_dam
        for j in range(len(ws_grid.columns)):
            tot_payout = calculate_payout(optimized_xs[j], optimized_ys[j], ws_grid, rp_dam_grid, nominal, j)
            payout_evt_grd.iloc[:,j] = tot_payout.iloc[:,0]
        tot_pay = np.sum(payout_evt_grd.iloc[i, :])
        if tot_pay > nominal:
                tot_pay = nominal
        else: 
            pass
        pay_dam_df.loc[i,"pay"] = tot_pay

    if include_plot:
        #Create a 1:1 plot (scatter plot with equal scales)
        plt.figure(figsize=(10, 10)) 
        plt.scatter(pay_dam_df['damage'], pay_dam_df['pay'], color='blue', marker='o', label='Events')

        # Add a 1:1 line for reference
        plt.plot([pay_dam_df['pay'].min(), pay_dam_df['pay'].max()], [pay_dam_df['pay'].min(), pay_dam_df['pay'].max()], color='red', linestyle='--', label='Trendline')

        #plot horizontal line at nominal
        plt.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 

        plt.xlabel('Damage [USD]')
        plt.ylabel('Payment [USD]')
        plt.title('1:1 Plot of Damage vs Payments for each event')
        plt.legend()
        plt.show()
    else: 
        pass

    return pay_dam_df

