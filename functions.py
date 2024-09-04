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
        #Create a DataFrame to sort the values and assign ranks
        df = df.sort_values(by='Damage', ascending=False)
        #Assign unique ranks
        df['Rank'] = range(1, len(df) + 1)
        #Map the ranks back to the original DataFrame
        #Sort the original DataFrame to match the original order
        df['RP'] = (r + 1)/df['Rank']
        df = df.sort_values(by='RP')
        #Extract sorted return periods and impacts
        sorted_rp = df['RP'].values
        sorted_damage = df['Damage'].values

        #Interpolate impacts for the given return periods
        calc_value = np.interp(return_period, sorted_rp, sorted_damage)
    else: 
        #Create a DataFrame to sort the values and assign ranks
        df = df.sort_values(by='pay', ascending=False)
        #Assign unique ranks
        df['Rank'] = range(1, len(df) + 1)
        #Map the ranks back to the original DataFrame
        #Sort the original DataFrame to match the original order
        df['RP'] = (r + 1)/df['Rank']
        df = df.sort_values(by='RP')
        #Extract sorted return periods and impacts
        sorted_rp = df['RP'].values
        sorted_pay = df['pay'].values

        #Interpolate impacts for the given return periods
        calc_value = np.interp(return_period, sorted_rp, sorted_pay)

    return calc_value

