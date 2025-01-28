'''Calibrate payout function by optimizing basis risk and compute payouts per event and return dataframe conatining all necessary information on each hazard event for the final bond simulations'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import minimize

#Define initial guesses for optimization
initial_guess_ws = [30, 40]  # Wind speed initial guess
initial_guess_cp = [990, 920]  # Central pressure initial guess

#this function calculates the payout for an event in a subarea -> defines the payout function
def init_alt_payout(min_trig, max_trig, haz_int, max_pay, int_haz_cp):
    intensities = np.array(haz_int.iloc[:, 0])
    payouts = np.zeros_like(intensities)
    if int_haz_cp:
        payouts[intensities <= max_trig] = max_pay
        mask = (intensities <= min_trig) & (intensities > max_trig)
        payouts[mask] = (intensities[mask] - min_trig) / (max_trig - min_trig) * max_pay

    else:
        payouts[intensities >= max_trig] = max_pay
        mask = (intensities >= min_trig) & (intensities < max_trig)
        payouts[mask] = (intensities[mask] - min_trig) / (max_trig - min_trig) * max_pay

    return payouts

#this function defines the objective function used in the optimization of the payout function to derive optimal minimum and maximum thresholds of the paramteric index
#returns the basis risk
def init_alt_objective_function(params, haz_int, damages, nominal, int_haz_cp):
    min_trig, max_trig = params
    max_dam = np.max(damages)
    if max_dam < nominal:
        max_pay = max_dam
    else: 
        max_pay = nominal
    payouts = init_alt_payout(min_trig, max_trig, haz_int, max_pay, int_haz_cp)
    if int_haz_cp:
        damage_per_grid = [float(damage) / integer if integer > 0 else 0
                           for damage, integer in zip(damages, np.array(haz_int['count_grids']))]
        arr_damages = np.array(damage_per_grid)
    else:
        arr_damages = np.array(damages)
    basis_risk = np.sum((arr_damages - payouts)**2)
    return basis_risk

#funtion to minimze basis risk by adjusting minimum and maximum parametric index thresholds used in the payout funciton
def init_alt_optimization(haz_int, nominal, damages_evt=None, damages_grid=None, print_params=True):
    grid_cells = range(len(haz_int.columns)-3)  
    grid_specific_results = {}
   
    if (haz_int.iloc[:, 0] > 900).any():
        initial_guess = initial_guess_cp
        int_haz_cp = True
    else:
        initial_guess = initial_guess_ws
        int_haz_cp = False

    results = {}
    for cell in grid_cells:

        damages = damages_evt if int_haz_cp else damages_grid.iloc[:,cell]

        # Perform optimization for each grid cell
        result = minimize(init_alt_objective_function, 
                          initial_guess, 
                          args=(haz_int.iloc[:,[cell, -1]], damages, nominal, int_haz_cp), 
                          method='COBYLA',
                          options={'maxiter': 100000})
        
        results[cell] = result

        if result.success:
            optimal_1, optimal_2 = result.x
            grid_specific_results[cell] = (optimal_1, optimal_2)
        else:
            print(f"Optimization failed for cell {cell}: {result.message}")

    if print_params:
        print(grid_specific_results)

    optimized_1 = np.array([values[0] for values in grid_specific_results.values()])  
    optimized_2 = np.array([values[1] for values in grid_specific_results.values()])  


    return results, optimized_1, optimized_2

#calculates the payout for each event in the dataframe and creates a dataframe containing damage, payout, year, and month for each event
#can create a plot optionally 
def alt_pay_vs_damage(damages_flt, optimized_1, optimized_2, haz_int, nominal, damages_grid=None, damages=None, exp=None):
    b = len(damages_flt)
    max_damage = damages_flt.max()
    if max_damage < 1:
        minimum_payout = 0
    else:
        minimum_payout = damages_flt[damages_flt > 0].min()
    payout_evt_grd = pd.DataFrame({letter: [None] * b for letter in haz_int.columns[:-2]})
    pay_dam_df = pd.DataFrame({'pay': [0.0] * b, 'damage': [0.0] * b, 'year': [0] * b, 'month': [0] * b})

    if (haz_int.iloc[:, 0] > 900).any():
        int_haz_cp = True
        max_dam = np.max(damages_flt)
    else:
        int_haz_cp = False

    for i in range(len(damages_flt)):
        tot_dam = damages_flt[i]
        pay_dam_df.loc[i,"damage"] = tot_dam
        pay_dam_df.loc[i,"year"] = int(haz_int['year'][i])
        pay_dam_df.loc[i,"month"] = int(haz_int['month'][i])
        for j in range(len(haz_int.columns)-3):
            grid_hazint = haz_int.iloc[:,[j, -1]]
            if int_haz_cp:
                None
            else:
                max_dam = np.max(damages_grid.iloc[:,j])
            if max_dam < nominal:
                max_pay = max_dam
            else: 
                max_pay = nominal
            payouts = init_alt_payout(optimized_1[j], optimized_2[j], grid_hazint, max_pay, int_haz_cp)
            payout_evt_grd.iloc[:,j] = payouts
        tot_pay = np.sum(payout_evt_grd.iloc[i, :])
        if tot_pay > nominal:
            tot_pay = nominal
        elif tot_pay < minimum_payout:
            tot_pay = 0
        else: 
            pass
        pay_dam_df.loc[i,"pay"] = tot_pay

    if damages is not None:
        tot_exp = exp.gdf['value'].sum()

        damages_df = pd.DataFrame(damages, columns=['Damage'])
        mask = damages_df['Damage'] <= nominal
        damages_df_flt = damages_df['Damage'][mask]
        payout_flt = pay_dam_df['pay'][mask]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        ax1.scatter(damages_df_flt/tot_exp, payout_flt/tot_exp, marker='o', color='blue', label='Events')
        ax1.plot([0, nominal/tot_exp], [0, nominal/tot_exp], color='black', linestyle='--', label='Trendline')
        ax1.axhline(y = nominal/tot_exp, color = 'r', linestyle = '-', label='Principal') 
        ax1.axhline(y = 0.05, color = 'r', linestyle = '-', label='Attachment Point') 
        ax1.axvline(x = 0.05, color = 'r', linestyle = '--', label='Min. Damage') 
        ax1.set_xlabel("Damage [share of GDP]", fontsize=12)
        ax1.set_ylabel("Payout [share of GDP]", fontsize=12)
        ax1.legend(loc='lower right', borderpad=2.0)

        ax2.scatter(damages/tot_exp, pay_dam_df['pay']/tot_exp, marker='o', color='blue', label='Events')
        ax2.axhline(y = nominal/tot_exp, color = 'r', linestyle = '-', label='Principal') 
        ax2.axhline(y = 0.05, color = 'r', linestyle = '-', label='Attachment Point') 
        ax2.axvline(x = 0.05, color = 'black', linestyle = '--', label='Min. Damage') 
        ax2.set_xscale('log')
        ax2.set_xlabel("Damage [share of GDP]", fontsize=12)
        ax2.set_ylabel("Payout [share of GDP]", fontsize=12)

        panel_labels = ["a)", "b)"]
        for i, ax in enumerate([ax1, ax2]):
            ax.annotate(panel_labels[i], 
                xy=(-0.1, 1),  
                xycoords="axes fraction", 
                fontsize=14, 
                fontweight="bold")
            
        plt.tight_layout()
        plt.show()

    else: 
        pass

    return pay_dam_df