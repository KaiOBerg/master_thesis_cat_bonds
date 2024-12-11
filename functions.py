import numpy as np
import pandas as pd
import calc_premium as cp

def check_scalar(variable):
    if np.isscalar(variable):
        cor_var = np.array([variable])
    else:
        cor_var = variable
    
    return cor_var

def get_all_values(d):
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.extend(get_all_values(value))
        else:
            values.append(value)
    return values

def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    :param iteration: Current iteration (int)
    :param total: Total iterations (int)
    :param prefix: Prefix string (str)
    :param suffix: Suffix string (str)
    :param decimals: Positive number of decimals in percent complete (int)
    :param length: Character length of bar (int)
    :param fill: Bar fill character (str)
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
    # Print New Line on Complete
    if iteration == total: 
        print()

def print_sng_bnd_rel_metr(bond_metrics, returns, premium_dic, nominal):
    print('Expected Loss:',round(premium_dic['exp_loss']*100, 1),'%')
    print('Attachment Probability:',round(premium_dic['att_prob']*100,1),'%')
    print('Coverage:',round(bond_metrics['Coverage'][0]*100,1),'%')
    print('Premium Ibrd:',round(premium_dic['ibrd']*100,1),'%;',round(premium_dic['ibrd']*nominal, 0),'USD' )
    print('Premium Chatoro et al.',round(premium_dic['regression']*100,1),'%;',round(premium_dic['regression']*nominal, 0),'USD')
    print('Premium Target Sharpe Ratio',round(premium_dic['required']*100,1),'%;',round(premium_dic['required']*nominal, 0),'USD')
    print('Standard Deviation of Returns',round(np.std(returns['Annual'][0]),2))

def print_mlt_bnd_rel_metr(countries, returns, premium_dic, tot_coverage_cty, nominal):
    print('Expected Loss: ',round(premium_dic['exp_loss']*100, 1),'%')
    print('Attachment Probability: ',round(premium_dic['att_prob']*100,1),'%')
    for i in countries:
        print(f'Coverage {i}:',round(tot_coverage_cty[i]['coverage']*100,1),'%')
    print('Premium Ibrd: ',round(premium_dic['ibrd']*100,1),'%; ',round(premium_dic['ibrd']*nominal, 0),'USD')
    print('Premium Chatoro et al.',round(premium_dic['regression']*100,1),'%; ',round(premium_dic['regression']*nominal, 0),'USD')
    print('Premium Target Sharpe Ratio',round(premium_dic['required']*100,1),'%; ',round(premium_dic['required']*nominal, 0),'USD')
    print('Standard Deviation Returns',np.std(returns['Annual'][0]))


def calc_rp_bnd_lss(ann_losses, return_period):
    """
    Compute impacts/payouts for specific return periods using a DataFrame.

    Parameters
    ----------
    df : pandas.Series
        A Series containing annual loss values.
    return_periods : Object
        The return period where we want to compute the exceedance impact/pay.
    damage : Boolean
        Indicating if function should return associated damage value or payout for given return period.

    Returns
    -------
    A number.
    """

    annual_losses = ann_losses['losses'].apply(sum)
    df = pd.DataFrame(annual_losses.sort_values(ascending=True))
    df['Rank'] = df.rank(method='first', ascending=False)
    df['RP'] = (len(df) + 1)/df['Rank']
    df = df.sort_values(by='RP')
    sorted_rp = df['RP'].values
    sorted_losses = df['losses'].values
    calc_value = np.interp(return_period, sorted_rp, sorted_losses)

    return calc_value

def create_tranches(rp_array, ann_losses):
    rows = []
    tranch_df = pd.DataFrame(columns=['RP', 'Loss'])
    for i in rp_array:
        loss = calc_rp_bnd_lss(ann_losses, i)
        rows.append({'RP': i, 'Loss': (loss)})
    rows.append({'RP': 'Max', 'Loss': (calc_rp_bnd_lss(ann_losses, 10000))})

    # Combine the rows into a DataFrame
    tranches = pd.concat([tranch_df, pd.DataFrame(rows)], ignore_index=True)
    tranches['nominal'] = 0.0
    tranches['nominal'] = tranches['Loss'].diff()
    tranches.at[0, 'nominal'] = tranches.at[0, 'Loss']
    el = 0
    tranches['expected_loss'] = 0.0
    tranches['expected_loss_own'] = 0.0
    tranches['lower_bound'] = 0.0
    tranches['upper_bound'] = 0.0
    tranches['premium'] = 0.0

    # Calculate lower and upper bounds, and expected loss
    for i in tranches.index:
        # Determine layer boundaries
        tranches.at[i, 'lower_bound'] = 0 if i == 0 else tranches.at[i - 1, 'Loss']
        tranches.at[i, 'upper_bound'] = tranches.at[i, 'Loss']

        # Losses within the tranche layer
        annual_losses = ann_losses['losses'].apply(sum)
        tranche_losses = (
            np.clip(annual_losses, tranches.at[i, 'lower_bound'], tranches.at[i, 'upper_bound']) 
            - tranches.at[i, 'lower_bound']
        )

        # Expected loss for the tranche
        tranche_el = np.mean(tranche_losses)
        el += tranche_el
        tranches.at[i, 'expected_loss'] = tranche_el

        tranches_loss_own =  np.array(tranche_losses)/tranches.at[i, 'nominal']
        tranches.at[i, 'expected_loss_own'] = np.mean(tranches_loss_own)
        
    # Calculate share of expected loss
    tranches['share_expected_loss'] = tranches['expected_loss'] / el

    tranches['nominal'] = tranches['Loss'].diff()
    tranches.at[0, 'nominal'] = tranches.at[0, 'Loss']

    tranches['premium'] = cp.calc_premium_regression(tranches['expected_loss_own'] *100)/100

    return tranches