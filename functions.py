import numpy as np
import pandas as pd

r = 10000

def calc_rp(df, return_period, damage=True):
    """
    Compute impacts/payouts for specific return periods using a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing impacts/payouts and their associated return periods.
        Should have columns 'Damage' and 'RP'.
    return_periods : Object
        The return period where we want to compute the exceedance impact.
    damage : Boolean
        Indicating if function should return associated damage value or payout for given return period.

    Returns
    -------
    pandas.DataFrame
        A List with the specified return period and their corresponding damage.
    """

    if damage == True:
        df['Rank'] = df['Damage'].rank(ascending=False, method='min')
        df['RP'] = (r + 1)/df['Rank']
        df = df.sort_values(by='RP')
        # Extract sorted return periods and impacts
        sorted_rp = df['RP'].values
        sorted_impact = df['Damage'].values

        # Interpolate impacts for the given return periods
        calc_value = np.interp(return_period, sorted_rp, sorted_impact)
    else: 
        df['Rank'] = df['pay'].rank(ascending=False, method='min')
        df['RP'] = (r + 1)/df['Rank']
        df = df.sort_values(by='RP')
        # Extract sorted return periods and impacts
        sorted_rp = df['RP'].values
        sorted_impact = df['pay'].values

        # Interpolate impacts for the given return periods
        calc_value = np.interp(return_period, sorted_rp, sorted_impact)


    return calc_value



#define maximum and minimum return periods to be covered
upper_rp = 200
lower_rp = 100


def init_up_low_cov(imp_grid_evt):
    rp_dam_grid = pd.DataFrame(columns=['Grid','Upper','Lower'])


    # Create an empty DataFrame
    for i in imp_grid_evt:
        number = ord(i) - 65
        df = pd.DataFrame({'Damage': imp_grid_evt[i]})
        df['Rank'] = df['Damage'].rank(ascending=False, method='min')
        df['RP'] = (10000 + 1)/df['Rank']
        df = df.sort_values(by='RP')
        rp_dam_grid.loc[number] = [i, calc_rp(df,upper_rp), calc_rp(df,lower_rp)]
        df_name = f'df_grid_{i}'
        globals()[df_name] = df
