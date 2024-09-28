import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.optimize import minimize

minimum_payout = 0.25

#Define bounds for minimum and maximum wind speeds
initial_guess_ws = [20, 61]  # Wind speed initial guess
bounds_ws = [(20, 40), (51, 250)]  # Bounds for wind speed thresholds
initial_guess_cp = [910, 960]  # Central pressure initial guess
bounds_cp = [(850, 944), (945, 1020)]  # Bounds for central pressure thresholds

def init_alt_payout(min_trig, max_trig, haz_int, nominal):
    payouts = []
    for i in range(len(haz_int)):
        payout = np.clip((haz_int.iloc[i, 0] - min_trig) / (max_trig - min_trig), 0, 1) * nominal
        if payout < (minimum_payout * nominal) or haz_int.iloc[i, 0] == 0:
            payout = 0
        payouts.append(payout)
    return payouts

def init_alt_objective_function(params, haz_int, damages, nominal):
    if (haz_int.iloc[:, 0] > 900).any():
        max_trig, min_trig = params
    else: 
        min_trig, max_trig = params
    payouts = init_alt_payout(min_trig, max_trig, haz_int, nominal)
    tot_payout = np.sum(payouts)
    damage_per_grid = [float(damage) / integer if integer > 0 else 0 
                       for damage, integer in zip(damages, np.array(haz_int['count_grids']))]
    tot_damages = np.sum(damage_per_grid)
    basis_risk = ((tot_damages - tot_payout)**2)**0.5
    return basis_risk

def init_alt_optimization(haz_int, damages, nominal, print_params=True):
    # Define bounds and initial guesses for each grid cell
    grid_cells = range(len(haz_int.columns)-2)  # Assuming 10 grid cells
    grid_specific_results = {}

    if (haz_int.iloc[:, 0] > 900).any():
        initial_guess = initial_guess_cp
        bounds = bounds_cp
    else:
        initial_guess = initial_guess_ws
        bounds = bounds_ws

    for cell in grid_cells:

        # Perform optimization for each grid cell
        result = minimize(init_alt_objective_function, 
                          initial_guess, 
                          args=(haz_int.iloc[:,[cell, -1]], damages, nominal), 
                          bounds=bounds, 
                          method='L-BFGS-B')

        if (haz_int.iloc[:, 0] > 900).any():
            optimal_max_int, optimal_min_int = result.x
        else: 
            optimal_min_int, optimal_max_int = result.x
        grid_specific_results[cell] = (optimal_min_int, optimal_max_int)

    if print_params:
        print(grid_specific_results)

    #Reshape parameters into a more interpretable form if needed
    optimized_xs = np.array([values[0] for values in grid_specific_results.values()])  #minimum threshold of intensity
    optimized_ys = np.array([values[1] for values in grid_specific_results.values()])  #maximum threshold of intensity

    return result, optimized_xs, optimized_ys

def alt_pay_vs_damage(damages, optimized_xs, optimized_ys, haz_int, nominal, include_plot=False):
    b = len(damages)
    payout_evt_grd = pd.DataFrame({letter: [None] * b for letter in haz_int.columns[:-2]})
    pay_dam_df = pd.DataFrame({'pay': [0.0] * b, 'damage': [0.0] * b, 'year': [0] * b})

    for i in range(len(damages)):
        tot_dam = damages[i]
        pay_dam_df.loc[i,"damage"] = tot_dam
        pay_dam_df.loc[i,"year"] = int(haz_int['year'][i])
        for j in range(len(haz_int.columns)-2):
            grid_hazint = haz_int.iloc[:,[j, -1]] 
            payouts = init_alt_payout(optimized_xs[j], optimized_ys[j], grid_hazint, nominal)
            payout_evt_grd.iloc[:,j] = payouts
        tot_pay = np.sum(payout_evt_grd.iloc[i, :])
        if tot_pay > nominal:
                tot_pay = nominal
        else: 
            pass
        pay_dam_df.loc[i,"pay"] = tot_pay

    if include_plot:

        mask = pay_dam_df['damage'] <= nominal
        damage_flt = pay_dam_df['damage'][mask]
        payout_flt = pay_dam_df['pay'][mask]

        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 6), gridspec_kw={'width_ratios': [10, 5, 5]})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        ax1.scatter(damage_flt, payout_flt, marker='o', color='blue', label='Events')
        ax1.plot([payout_flt.min(), payout_flt.max()], [payout_flt.min(), payout_flt.max()], color='red', linestyle='--', label='Trendline')
        ax1.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 

        # Add labels and title
        ax1.set_title("Damage vs. Payout for each Event")
        ax1.set_xlabel("Damage [USD]")
        ax1.set_ylabel("Payout [USD]")
        ax1.legend(loc='upper left', borderpad=2.0)

        # Create an inset plot (overview of total data)
        inset_ax1 = inset_axes(ax1, width="30%", height="30%", loc='lower right', borderpad=3.0)  # adjust size and position
        inset_ax1.scatter(pay_dam_df['damage'], pay_dam_df['pay'], label='Overview Data', marker='o', color='blue')
        inset_ax1.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        #inset_ax1.set_xscale('log')
        #inset_ax1.set_yscale('log')
        inset_ax1.set_xlabel("Damage [USD]", fontsize=8)
        inset_ax1.set_ylabel("Payout [USD]", fontsize=8)

        ax2.scatter(damage_flt, payout_flt, marker='o', color='blue', label='Events')
        ax2.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        ax2.set_xscale('log')
        # Add labels and title
        ax2.set_title("Damage vs. Payout for each Event - Low Damages")
        ax2.set_xlabel("Damage [USD]")
        ax2.set_ylabel("Payout [USD]")
        ax2.legend()


        # Create an inset plot (overview of total data)
        inset_ax2 = inset_axes(ax2, width="30%", height="30%", loc='upper left', borderpad=4.0)  # adjust size and position
        inset_ax2.scatter(damage_flt, payout_flt, label='Overview Data', marker='o', color='blue')
        inset_ax2.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        inset_ax2.set_xlabel("Damage [USD]", fontsize=8)
        inset_ax2.set_ylabel("Payout [USD]", fontsize=8)
        # Show both plots
        plt.tight_layout()
        plt.show()

    else: 
        pass

    return pay_dam_df