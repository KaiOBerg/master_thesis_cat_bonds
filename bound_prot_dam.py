'''Script used to calculate return periods and filter events so that they meet the minimum payout criteria'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import set_nominal as snom

r = 10000 #number of simulated years in tc dataset


'''calculate return beriods for damages or payouts'''
def calc_rp(df, return_period, damage=True):
    """
    Compute impacts/payouts for specific return periods using a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing impacts/payouts and their associated return periods.
        Should have columns 'Damage'or 'Pay and 'RP'.
    return_periods : Object
        The return period where we want to compute the exceedance impact/pay.
    damage : Boolean
        Indicating if function should return associated damage value or payout for given return period.

    Returns
    -------
    A number.
    """

    if damage == True:
        calc_rp_for = 'Damage'
    else:
        calc_rp_for = 'pay'
    #Create a DataFrame to sort the values and assign ranks
    df = df.sort_values(by=calc_rp_for, ascending=False)
    #Assign unique ranks
    df['Rank'] = range(1, len(df) + 1)
    #Map the ranks back to the original DataFrame
    #Sort the original DataFrame to match the original order
    df['RP'] = (r + 1)/df['Rank']
    df = df.sort_values(by='RP')
    #Extract sorted return periods and impacts
    sorted_rp = df['RP'].values
    sorted_damage = df[calc_rp_for].values
    #Interpolate impacts for the given return periods
    calc_value = np.interp(return_period, sorted_rp, sorted_damage)


    return calc_value

'''calculate return period and filter damage data for events with damage greater than minimum payout'''
def init_imp_flt(imp_per_event, imp_admin_evt, lower_rp=None, prot_share=None, exposure=None):
    imp_per_event_df = pd.DataFrame({'Damage': imp_per_event})
    imp_per_event_flt=np.array(imp_per_event_df)
    if lower_rp is not None:
        to_protect = calc_rp(imp_per_event_df, lower_rp, damage=True)
    else: 
        to_protect = snom.init_nominal(exposure=exposure, prot_share=prot_share, print_nom=False)

    imp_per_event_flt[imp_per_event_flt < to_protect] = 0
    imp_admin_evt_flt = imp_admin_evt.copy()
    imp_admin_evt_flt.loc[imp_admin_evt_flt.sum(axis=1) < to_protect, :] = 0


    return imp_per_event_flt, imp_admin_evt_flt, to_protect


'''not used for final results'''
def init_dam_ret_per_grid(imp_grid_evt, lower_rp, adj_max=False, nominal=None, plt_dam_rp_grid=None):
    """
    Initializes the return period grid by calculating damages per grid cell and their corresponding return periods.
    
    Args:
        imp_grid_evt (dict): A dictionary where keys are grid identifiers (letters) and values are damages per event.
        lower_rp (float): A parameter used for calculating the lower return period damage.
        nominal (float): Nominal of the bond which can be used as maximum boundary of to be protected damages.
        adj_max (bool): Flag to adjust the maximum damage. Default is False.
        plt_dam_rp_grid (str or None): Grid identifier to plot. Default is set to None to skip plotting. If plot is desired input must correspond to grid letter e.g. 'A'.

    Returns:
        tuple: A tuple containing:
            - rp_dam_grid (pd.DataFrame): DataFrame with grid identifiers and calculated damages for the specified return periods.
            - dam_rp_per_grid (dict): Dictionary containing DataFrames of damages with return periods per grid cell.
            - imp_grid_evt_flt: Adjusted impact grid event (filtered).
    """
    #Create an empty DataFrame
    rp_dam_grid = pd.DataFrame(columns=['Grid','Lower'])

    dam_rp_per_grid = {} #empty dictionary where to save dataframes of damage with return periods per grid cell
    #loop through damages per event
    for i in imp_grid_evt:
        number = ord(i) - 65 #transform grid letter to number
        df = pd.DataFrame({'Damage': imp_grid_evt[i]}) #get damages per grid
        df['Rank'] = df['Damage'].rank(ascending=False, method='min') #rank damages
        df['RP'] = (r + 1)/df['Rank'] #calcualte return period per damage
        df = df.sort_values(by='RP') 
        dam_rp_per_grid[i] = df #save damages per grid cell in dictionary
        rp_dam_grid.loc[number] = [i, calc_rp(df,lower_rp)] #calculate damage of certain return periods, set max damage to nominal

    if plt_dam_rp_grid is not None:

        mask = dam_rp_per_grid[plt_dam_rp_grid]['RP'] < 500
        return_period_flt = dam_rp_per_grid[plt_dam_rp_grid]['RP'][mask]
        impact_flt = dam_rp_per_grid[plt_dam_rp_grid]['Damage'][mask]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 6), gridspec_kw={'width_ratios': [10, 5, 5]})

        ax1.plot(return_period_flt, impact_flt, label='Filtered Data', linestyle='-', color='red')

        ax1.set_title(f'Exceedance frequency curve - Grid {plt_dam_rp_grid}')
        ax1.set_xlabel("Return Period [Years]")
        ax1.set_ylabel("Impact [USD]")

        inset_ax1 = inset_axes(ax1, width="30%", height="30%", loc='upper left', borderpad=3.0)  
        inset_ax1.plot(dam_rp_per_grid[plt_dam_rp_grid]['RP'], dam_rp_per_grid[plt_dam_rp_grid]['Damage'], label='Overview Data', linestyle='-', color='red')
        inset_ax1.set_xlabel("Return Period [Years]", fontsize=8)
        inset_ax1.set_ylabel("Impact [USD]", fontsize=8)

        ax2.plot(dam_rp_per_grid[plt_dam_rp_grid]['RP'], dam_rp_per_grid[plt_dam_rp_grid]['Damage'], linestyle='-', color='red')
        ax2.set_xscale('log')
        ax2.set_xlabel('Return Period [Years]')
        ax2.set_ylabel('Impact [USD]')
        ax2.set_title(f'Exceedance frequency curve - Grid {plt_dam_rp_grid} - Log')

        ax3.plot(dam_rp_per_grid[plt_dam_rp_grid]['RP'], dam_rp_per_grid[plt_dam_rp_grid]['Damage'], linestyle='-', color='red')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Return Period [Years]')
        ax3.set_ylabel('Impact [USD]')
        ax3.set_title(f'Exceedance frequency curve - Grid {plt_dam_rp_grid} - LogLog')

        plt.tight_layout()
        plt.show()

    imp_grid_evt_flt = adj_imp_grid_evt(imp_grid_evt, rp_dam_grid, adj_max=False, nominal=None)

    return rp_dam_grid, dam_rp_per_grid, imp_grid_evt_flt

'''sets damages to 0 if there below minimum payout and if wanted sets damages greater than the principal to the principal -> not used for final results'''
def adj_imp_grid_evt(imp_grid_evt, rp_dam_grid, adj_max=False, nominal=None):

    imp_grid_evt_flt = imp_grid_evt.copy()

    #the current implimitation sets damages below the lower return period to zero and depending on input sets max damage to nominal
    for i in range(len(rp_dam_grid)):
        min_val = rp_dam_grid.loc[i,'Lower']
        for j in range(len(imp_grid_evt)):
            sel_val = imp_grid_evt.iloc[j,i]
            if adj_max and sel_val > nominal:
                imp_grid_evt_flt.iloc[j,i] = nominal
            if sel_val < min_val:
                imp_grid_evt_flt.iloc[j,i] = 0
            else:
                pass
    
    return imp_grid_evt_flt