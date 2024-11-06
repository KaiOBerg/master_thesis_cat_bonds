import numpy as np

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

