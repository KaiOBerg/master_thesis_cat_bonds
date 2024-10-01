import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import functions as fct


term = 3
simulated_years = 10001


def init_bond_exp_loss(events_per_year, nominal, nominal_lst_cty=None):
    ann_loss = []
    cur_nominal = nominal
    if nominal_lst_cty is not None:
        cur_nom_cty = nominal_lst_cty
    else:
        cur_nom_cty = [1,1]

    payout_count = 0
    cty_losses = {k: [] for k in range(len(events_per_year))}

    for k in range(term):
        losses = []
        for q in range(len(events_per_year)):
            payouts = np.sum(events_per_year[q][k]['pay'].to_numpy())
            #If there are events in the year, sample that many payouts and the associated damages
            if payouts == 0 or cur_nominal == 0 or cur_nom_cty[q] == 0:
                sum_payouts = 0
            elif payouts > 0:
                sum_payouts = payouts 
                if nominal_lst_cty is not None:
                    cur_nom_cty[q] -= sum_payouts
                    if cur_nom_cty[q] < 0:
                        sum_payouts += cur_nom_cty[q]
                        cur_nom_cty[q] = 0
                cur_nominal -= sum_payouts
                if cur_nominal < 0:
                    sum_payouts += cur_nominal
                    cur_nominal = 0
                else:
                    pass
            
            losses.append(sum_payouts)
            cty_losses[q].append(sum_payouts)
        ann_pay = np.sum(losses)
        if ann_pay > 0:
            payout_count += 1
        ann_loss.append(ann_pay)

    att_prob = payout_count / term
    rel_tot_loss = np.sum(ann_loss) / nominal
    rel_losses = list(np.array(ann_loss) / nominal)
    for key in cty_losses:
        cty_losses[key] = [cty_abs_loss / nominal for cty_abs_loss in cty_losses[key]]

    return rel_losses, att_prob, rel_tot_loss, cty_losses



def init_exp_loss_att_prob_simulation(pay_dam_df_dic, nominal, nominal_lst_cty=None, print_prob=True):
    att_prob_list = []
    annual_losses = []
    total_losses = []
    ann_cty_losses = {u: [] for u in range(len(pay_dam_df_dic))}

    for i in range(simulated_years-3):
        events_per_year = {u: [] for u in range(len(pay_dam_df_dic))}
        for j in range(term):
            keys = list(pay_dam_df_dic.keys())
            for r in range(len(pay_dam_df_dic)):
                if 'year' in pay_dam_df_dic[keys[r]].columns:
                    events_per_year[r].append(pay_dam_df_dic[keys[r]][pay_dam_df_dic[keys[r]]['year'] == (i+j)])
                else:
                    events_per_year[r].append(pd.DataFrame({'pay': [0], 'damage': [0]}))
        losses, att_prob, rel_tot_loss, cty_losses = init_bond_exp_loss(events_per_year, nominal, nominal_lst_cty)

        att_prob_list.append(att_prob)
        annual_losses.extend(losses)
        total_losses.append(rel_tot_loss)

        for key in cty_losses:
            ann_cty_losses[key].extend(cty_losses[key])

    # Convert simulated net cash flows to a series
    att_prob = np.mean(att_prob_list)
    exp_loss_ann = np.mean(annual_losses)

    annual_losses = pd.Series(annual_losses)
    total_losses = pd.Series(total_losses)

    VaR_99_ann = annual_losses.quantile(0.99)
    VaR_99_tot = total_losses.quantile(0.99)
    VaR_95_ann = annual_losses.quantile(0.95)
    VaR_95_tot = total_losses.quantile(0.95)
    ES_99_ann = annual_losses[annual_losses > VaR_99_ann].mean()
    ES_99_tot = total_losses[total_losses > VaR_99_tot].mean()
    ES_95_ann = annual_losses[annual_losses > VaR_95_ann].mean()
    ES_95_tot = total_losses[total_losses > VaR_95_tot].mean()
    MES_cty = {country: {'95': None, '99': None} for country in ann_cty_losses.keys()}
    for country, ann_cty_losses_iter in ann_cty_losses.items():
        ann_cty_losses_iter = pd.Series(ann_cty_losses_iter)
        MES_cty[country]['95'] = ann_cty_losses_iter[annual_losses > VaR_95_ann].mean()
        MES_cty[country]['99'] = ann_cty_losses_iter[annual_losses > VaR_99_ann].mean()

    es_metrics = {'VaR_99_ann': VaR_99_ann, 'VaR_99_tot': VaR_99_tot, 'VaR_95_ann': VaR_95_ann, 'VaR_95_tot': VaR_95_tot,
                  'ES_99_ann': ES_99_ann, 'ES_99_tot': ES_99_tot, 'ES_95_ann': ES_95_ann, 'ES_95_tot': ES_95_tot}
    
    if print_prob:
        print(f'Expected Loss = {exp_loss_ann}')
        print(f'Attachment Probability = {att_prob}')

    return exp_loss_ann, att_prob, annual_losses, total_losses, es_metrics, MES_cty

def init_bond_simulation(pay_dam_df_dic, premiums, rf_rate, nominal, countries, want_ann_returns=True, model_rf=False):

    metric_names = ['tot_payout', 'tot_damage', 'tot_pay']

    #Check if premiums/rf_rates are single values
    premiums = fct.check_scalar(premiums)

    metrics_per_premium = pd.DataFrame(index = range(len(premiums)), columns=["Premium", "Sharpe_ratio_ann", "Sharpe_ratio_tot",
                                                                              "Coverage", "Basis_risk", "Average Payments", "Summed Payments"])

    returns_per_premium = pd.DataFrame(index = range(len(premiums)), columns=["Premium","Annual", "Total"])

    tot_coverage_prem_cty = {}
    for code in countries:
        tot_coverage_prem_cty[code] = {'payout': [], 'damage': [], 'coverage': [], 'count_zero': 0}

    for z, premium in enumerate(premiums):
        annual_returns = []
        tot_returns = []
        rf_annual = []
        rf_total = []
        tot_coverage_cty = {}
        for code in countries:
            tot_coverage_cty[code] = {'payout': [], 'damage': [], 'count_zero': 0}
        metrics_sim = {key: [] for key in metric_names}
        for i in range(simulated_years-3):
            #model interest rates if wanted
            if model_rf:
                rf = init_model_rf(rf_rate)
            else:
                rf = rf_rate
            events_per_year = {u: [] for u in range(len(pay_dam_df_dic))}
            for j in range(term):
                keys = list(pay_dam_df_dic.keys())
                for r in range(len(pay_dam_df_dic)):
                    if 'year' in pay_dam_df_dic[keys[r]].columns:
                        events_per_year[r].append(pay_dam_df_dic[keys[r]][pay_dam_df_dic[keys[r]]['year'] == (i+j)])
                    else:
                        events_per_year[r].append(pd.DataFrame({'pay': [0], 'damage': [0]}))
            simulated_ncf_rel, metrics, rf_rates_list, coverage_cty = init_bond(events_per_year, premium, rf, nominal, countries)
    
            metrics_sim['tot_payout'].append(metrics['tot_payout'])
            metrics_sim['tot_damage'].append(metrics['tot_damage'])
            metrics_sim['tot_pay'].append(metrics['tot_pay'])
            for key in coverage_cty.keys():
                tot_coverage_cty[key]['payout'].append(coverage_cty[key]['payout'])
                if coverage_cty[key]['payout'] == 0:
                    tot_coverage_cty[key]['count_zero'] += 1
                tot_coverage_cty[key]['damage'].append(coverage_cty[key]['damage'])

            if want_ann_returns:
                annual_returns.extend(simulated_ncf_rel)

            else:
                ann_return = (1 + sum(simulated_ncf_rel)) ** (1/term) - 1
                annual_returns.append(ann_return)

            tot_returns.append(np.sum(simulated_ncf_rel))
            rf_annual.append(np.mean(rf_rates_list))
            rf_total.append(np.sum(rf_rates_list))

        # Convert simulated net cash flows to a series
        annual_returns = pd.Series(annual_returns)
        tot_returns = pd.Series(tot_returns)
        #calculate finacial metrics
        metrics_sim_sum = {}
        metrics_sim_sum['tot_payout'] = np.sum(metrics_sim['tot_payout'])
        metrics_sim_sum['tot_damage'] = np.sum(metrics_sim['tot_damage'])
        metrics_sim_sum['tot_pay'] = np.nanmean(metrics_sim['tot_pay'])
        for key in tot_coverage_cty.keys():
            tot_coverage_prem_cty[key]['coverage'].append(sum(tot_coverage_cty[key]['payout']) / sum(tot_coverage_cty[key]['damage']))
            tot_coverage_prem_cty[key]['payout'].append(sum(tot_coverage_cty[key]['payout']))
            tot_coverage_prem_cty[key]['damage'].append(sum(tot_coverage_cty[key]['damage']))
            tot_coverage_prem_cty[key]['count_zero'] += tot_coverage_cty[key]['count_zero']

        premium_float = np.float64(premium)

        sharpe_ratio_ann = init_sharpe_ratio(annual_returns, rf_annual)
        sharpe_ratio_tot = init_sharpe_ratio(tot_returns, rf_total)

        metrics_per_premium.loc[z] = [premium_float, sharpe_ratio_ann, sharpe_ratio_tot,
                                      metrics_sim_sum['tot_payout']/metrics_sim_sum['tot_damage'], metrics_sim_sum['tot_payout']-metrics_sim_sum['tot_damage'], metrics_sim_sum['tot_pay'], metrics_sim_sum['tot_payout']]  
        
        returns_per_premium.loc[z] = [premium_float, annual_returns, tot_returns]

            
    return metrics_per_premium, returns_per_premium, tot_coverage_prem_cty

def init_bond(events_per_year, premium, risk_free_rates, nominal, countries):   
    simulated_ncf = []
    tot_payout = []
    tot_damage = []
    coverage_cty = {}
    for code in countries:
        coverage_cty[code] = {'payout': [], 'damage': []}
    rf_rates_list = []
    metrics = {}    
    cur_nominal = nominal
    country_payouts = {code: [] for code in countries}
    country_damages = {code: [] for code in countries}

    for k in range(term):
        rf = check_rf(risk_free_rates, k)
        rf_rates_list.append(rf)
        net_cash_flow = 0
        country_tot_payouts = []
        country_tot_damages = []
        for q in range(len(events_per_year)):
            #randomly generate number of events in one year using poisson distribution and calculated yearly event probability
            payouts = np.sum(events_per_year[q][k]['pay'].to_numpy()) 
            damages = np.sum(events_per_year[q][k]['damage'].to_numpy())  
            #If there are events in the year, sample that many payouts and the associated damages
            sum_damages = damages
            if payouts == 0:
                sum_payouts = 0
            elif payouts > 0:
                sum_payouts = payouts
                cur_nominal -= sum_payouts
                if cur_nominal < 0:
                    sum_payouts += cur_nominal
                    cur_nominal = 0
                else:
                    pass
            
            country_payouts[countries[q]].append(sum_payouts)
            country_damages[countries[q]].append(sum_damages)
            country_tot_payouts.append(sum_payouts)
            country_tot_damages.append(sum_damages)

        ann_pay = np.sum(country_tot_payouts)

        if ann_pay == 0:
            net_cash_flow = cur_nominal * (premium + rf)
        else: 
            net_cash_flow = (cur_nominal * (premium + rf)) - ann_pay

        tot_payout.append(ann_pay)
        simulated_ncf.append(net_cash_flow)
        tot_damage.append(np.sum(country_tot_damages))

    simulated_ncf_rel = list(np.array(simulated_ncf) / nominal)
    metrics['tot_payout'] = np.sum(tot_payout)
    metrics['tot_damage'] = np.sum(tot_damage)
    for key in country_payouts.keys():
        coverage_cty[key]['payout'] = sum(country_payouts[key])
        coverage_cty[key]['damage'] = sum(country_damages[key])
    if np.sum(tot_payout) == 0:
        tot_pay = np.nan
    else:
        tot_pay = np.sum(tot_payout)
    metrics['tot_pay'] = tot_pay

    return simulated_ncf_rel, metrics, rf_rates_list, coverage_cty

def init_requ_premium(requ_sharpe_ratio, simulation_matrix, rf_rates):

    # Define the difference function between Sharpe ratio curve and required Sharpe ratio
    def intersection_func(x):
        return np.float64(sharpe_interp(x).item()) - requ_sharpe_ratio
    
    requ_premiums = {}

    rf_rates = fct.check_scalar(rf_rates)

    for rf in rf_rates:
        rf_str = str(rf)
        premiums = simulation_matrix[rf_str]['Premium']
        sharpe_ratios = simulation_matrix[rf_str]['Sharpe_ratio_ann']

        # Interpolate the Sharpe ratio curve
        sharpe_interp = interp1d(premiums, sharpe_ratios, kind='linear')

        # Use fsolve to find the intersection point(s), provide a guess
        x_guess = 0.01  # Initial guess based on the range of premiums
        x_intersection = fsolve(intersection_func, x_guess)[0]

        # Calculate the corresponding Sharpe ratio at the intersection point
        y_intersection = sharpe_interp(x_intersection)

        requ_premiums[rf_str] = x_intersection

        print(f"Intersection point using risk free interest rate of {rf*100}%: Premium = {x_intersection:.4f}, Sharpe Ratio = {y_intersection:.4f}")

    return requ_premiums

def display_premiums(requ_premiums, requ_sharpe_ratio, rf_rates, simulated_metrics):
    rf_rates = fct.check_scalar(rf_rates)
    for rf in rf_rates:
        rf_str = str(rf)
        risk_multiple = requ_premiums[rf_str]/simulated_metrics[rf_str]['Annual_expected_loss'][1]
        
        plt.plot(simulated_metrics[rf_str]['Premium'], simulated_metrics[rf_str]['Sharpe_ratio_ann'])
        plt.axhline(y = requ_sharpe_ratio, color = 'r', linestyle = '-', label='Required Sharpe Ratio') 
        plt.text(0.1, 0.4, f'Risk free rate: {round(rf*100,3)}%', fontsize = 12)
        plt.text(0.1, 0.3, f'Required Premium: {round(requ_premiums[rf_str]*100,3)}%', fontsize = 12)
        plt.text(0.1, 0.2, f'Risk Multiple: {round(risk_multiple,3)}', fontsize = 12)
        plt.xlabel('Premium [share of nominal]')
        plt.ylabel('Sharpe ratio')
        plt.show()

def init_sharpe_ratio(rel_returns, risk_free_rate, exp_short=None):
    exp_ret = np.mean(rel_returns)
    rf = np.mean(risk_free_rate)
    if exp_short: 
        std = exp_short
    else:
        std = np.std(rel_returns)
    sharpe_ratio = (exp_ret - rf) / std
    return sharpe_ratio

def init_coverage(payouts, damages):
    if np.sum(payouts) == np.sum(damages) == 0:
        coverage = np.nan
        basis_risk = np.nan
        tot_pay = np.nan
    elif np.sum(damages) == 0: 
        coverage = 1
        basis_risk = np.sum(payouts)
        tot_pay = np.sum(payouts)
    else:
        coverage = np.sum(payouts) / np.sum(damages)
        basis_risk = np.sum(payouts) - np.sum(damages)
        tot_pay = np.sum(payouts)

    return coverage, basis_risk, tot_pay

def init_expected_loss(returns):
    #Filter to get the returns that are negative -> losses
    losses = returns.apply(lambda x: 0 if x > 0 else x)
    loss_magnitudes = -losses
    expected_loss = np.mean(loss_magnitudes)
    return expected_loss

def init_model_rf(risk_free_rate):

    modeled_rf_rates = [risk_free_rate] * term

    return modeled_rf_rates

def check_rf(risk_free_rates, iterator):

    if isinstance(risk_free_rates, list):
        rf = risk_free_rates[iterator]
    else:
        rf = risk_free_rates

    return rf