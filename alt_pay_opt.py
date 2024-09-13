import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

minimum_payout = 0.1

#Define bounds for minimum and maximum wind speeds
initial_guess = [20, 61]  # Example initial guess
bounds = [(20, 40), (51, 150)]  # Bounds for thresholds

def init_alt_payout(min_speed, max_speed, haz_int, nominal):
    payouts = []
    for i in range(len(haz_int)):
        payout = np.clip((haz_int[i] - min_speed) / (max_speed - min_speed), 0, 1) * nominal
        if payout < (minimum_payout * nominal):
            payout = 0
        payouts.append(payout)
    return payouts


def init_alt_objective_function(params, haz_int, damages, nominal):
    min_speed, max_speed = params
    payouts = init_alt_payout(min_speed, max_speed, haz_int, nominal)
    tot_payout = np.sum(payouts)
    tot_damages = np.sum(damages)
    basis_risk = ((tot_damages - tot_payout)**2)**0.5
    return basis_risk

def init_alt_optimization(haz_int, damages, nominal, print_params=True):
    # Define bounds and initial guesses for each grid cell
    grid_cells = range(len(haz_int.columns))  # Assuming 10 grid cells
    grid_specific_results = {}

    for cell in grid_cells:

        # Perform optimization for each grid cell
        result = minimize(init_alt_objective_function, 
                          initial_guess, 
                          args=(haz_int.iloc[:,cell], damages.iloc[:,cell], nominal), 
                          bounds=bounds, 
                          method='L-BFGS-B')

        optimal_min_speed, optimal_max_speed = result.x
        grid_specific_results[cell] = (optimal_min_speed, optimal_max_speed)

    if print_params:
        print(grid_specific_results)

    #Reshape parameters into a more interpretable form if needed
    optimized_xs = np.array([values[0] for values in grid_specific_results.values()])  #minimum threshold of windspeeds
    optimized_ys = np.array([values[1] for values in grid_specific_results.values()])  #maximum threshold of windspeeds

    return result, optimized_xs, optimized_ys

def alt_pay_vs_damage(damages, optimized_xs, optimized_ys, haz_int, nominal, include_plot=False):
    b = len(damages)
    payout_evt_grd = pd.DataFrame({letter: [None] * b for letter in haz_int.columns})
    pay_dam_df = pd.DataFrame({'pay': [0.0] * b, 'damage': [0.0] * b})

    for i in range(len(damages)):
        tot_dam = np.sum(damages.iloc[i, :])
        pay_dam_df.loc[i,"damage"] = tot_dam
        for j in range(len(haz_int.columns)):
            grid_hazint = haz_int.iloc[:,j] 
            payouts = init_alt_payout(optimized_xs[j], optimized_ys[j], grid_hazint, nominal)
            payout_evt_grd.iloc[:,j] = payouts
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