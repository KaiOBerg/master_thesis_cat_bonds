import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.optimize import minimize

minimum_payout = 0.25

#Define bounds for minimum and maximum wind speeds
initial_guess_ws = [30, 45, 60]  # Wind speed initial guess
initial_guess_cp = [980, 940, 915]  # Central pressure initial guess

def init_alt_payout(step1, step2, step3, haz_int, nominal, int_haz_cp):
    payouts = []

    if int_haz_cp:
        for i in range(len(haz_int)):
            if haz_int.iloc[i, 0] == 0:
                payout = 0
            elif  haz_int.iloc[i, 0] < step3:
                payout = nominal
            elif  haz_int.iloc[i, 0] < step2:
                payout = nominal * 0.7
            elif  haz_int.iloc[i, 0] < step1:
                payout = nominal * 0.3
            else: 
                payout = 0
            payouts.append(payout)
    else: 
        for i in range(len(haz_int)):
            if haz_int.iloc[i, 0] == 0:
                payout = 0
            elif  haz_int.iloc[i, 0] > step3:
                payout = nominal
            elif  haz_int.iloc[i, 0] > step2:
                payout = nominal * 0.7
            elif  haz_int.iloc[i, 0] > step1:
                payout = nominal * 0.3
            else: 
                payout = 0
            payouts.append(payout)

    return payouts

def init_alt_objective_function(params, haz_int, damages, nominal, int_haz_cp):
    step1, step2, step3 = params
    payouts = init_alt_payout(step1, step2, step3, haz_int, nominal, int_haz_cp)
    tot_payout = np.sum(payouts)
    if int_haz_cp:
        damage_per_grid = [float(damage) / integer if integer > 0 else 0
                           for damage, integer in zip(damages, np.array(haz_int['count_grids']))]
        tot_damages = np.sum(damage_per_grid)
    else:
        tot_damages = np.sum(damages)
    basis_risk = ((tot_damages - tot_payout)**2)**0.5
    return basis_risk

def init_alt_optimization(haz_int, nominal, damages_evt=None, damages_grid=None, print_params=True):
    # Define bounds and initial guesses for each grid cell
    grid_cells = range(len(haz_int.columns)-2)  
    grid_specific_results = {}
    
    if (haz_int.iloc[:, 0] > 900).any():
        initial_guess = initial_guess_cp
        int_haz_cp = True
    else:
        initial_guess = initial_guess_ws
        int_haz_cp = False

    results = {}
    for cell in grid_cells:

        cons = init_cons(int_haz_cp)
        if int_haz_cp:
            damages = damages_evt
        else: 
            damages = damages_grid.iloc[:,cell]

        # Perform optimization for each grid cell
        result = minimize(init_alt_objective_function, 
                          initial_guess, 
                          args=(haz_int.iloc[:,[cell, -1]], damages, nominal, int_haz_cp), 
                          method='COBYLA',
                          constraints=cons)
        
        results[cell] = result

        if result.success:
            optimal_step1, optimal_step2, optimal_step3 = result.x
            grid_specific_results[cell] = (optimal_step1, optimal_step2, optimal_step3)
        else:
            print(f"Optimization failed for cell {cell}: {result.message}")

    if print_params:
        print(grid_specific_results)

    #Reshape parameters into a more interpretable form if needed
    optimized_step1 = np.array([values[0] for values in grid_specific_results.values()])  
    optimized_step2 = np.array([values[1] for values in grid_specific_results.values()])  
    optimized_step3 = np.array([values[2] for values in grid_specific_results.values()])  


    return result, optimized_step1, optimized_step2, optimized_step3

def cons_ws_1(params):
    return params[1] - params[0]  
def cons_ws_2(params):
    return params[2] - params[0]  
def cons_ws_3(params):
    return params[2] - params[1]
def cons_cp_1(params):
    return params[0] - params[1]  
def cons_cp_2(params):
    return params[0] - params[2]  
def cons_cp_3(params):
    return params[1] - params[2]

def init_cons(int_haz_cp):
    if int_haz_cp:
        cons = [{'type': 'ineq', 'fun': cons_cp_1},
                     {'type': 'ineq', 'fun': cons_cp_2},
                     {'type': 'ineq', 'fun': cons_cp_3}]
    else:
        cons = [{'type': 'ineq', 'fun': cons_ws_1},
                {'type': 'ineq', 'fun': cons_ws_2},
                {'type': 'ineq', 'fun': cons_ws_3}]

    return cons

def alt_pay_vs_damage(damages, optimized_step1, optimized_step2, optimized_step3, haz_int, nominal, include_plot=False):
    b = len(damages)
    payout_evt_grd = pd.DataFrame({letter: [None] * b for letter in haz_int.columns[:-2]})
    pay_dam_df = pd.DataFrame({'pay': [0.0] * b, 'damage': [0.0] * b, 'year': [0] * b})

    if (haz_int.iloc[:, 0] > 900).any():
        int_haz_cp = True
    else:
        int_haz_cp = False

    for i in range(len(damages)):
        tot_dam = damages[i]
        pay_dam_df.loc[i,"damage"] = tot_dam
        pay_dam_df.loc[i,"year"] = int(haz_int['year'][i])
        for j in range(len(haz_int.columns)-2):
            grid_hazint = haz_int.iloc[:,[j, -1]] 
            payouts = init_alt_payout(optimized_step1[j], optimized_step2[j], optimized_step3[j], grid_hazint, nominal, int_haz_cp)
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
        ax1.set_xscale('log')

        # Add labels and title
        ax1.set_title("Damage vs. Payout - Damage <= Nominal")
        ax1.set_xlabel("Damage [USD]")
        ax1.set_ylabel("Payout [USD]")
        ax1.legend(loc='upper left', borderpad=2.0)

        ax2.scatter(pay_dam_df['damage'], pay_dam_df['pay'], marker='o', color='blue', label='Events')
        ax2.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        ax2.set_xscale('log')
        # Add labels and title
        ax2.set_title("Damage vs. Payout for each Event <= Nominal - Log")
        ax2.set_xlabel("Damage [USD]")
        ax2.set_ylabel("Payout [USD]")
        ax2.legend(loc='upper left', borderpad=2.0)

        # Show both plots
        plt.tight_layout()
        plt.show()

    else: 
        pass

    return pay_dam_df