import pandas as pd
import numpy as np
import random
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt

import functions as fct


term = 3
simulated_years = 10001

def init_bond_exp_loss(events_per_year, nominal):
    losses = []
    cur_nominal = nominal
    payout_count = 0

    for k in range(term):
        #randomly generate number of events in one year using poisson distribution and calculated yearly event probability
        payouts = np.sum(events_per_year[k]['pay'].to_numpy())
        #If there are events in the year, sample that many payouts and the associated damages
        if payouts == 0 or cur_nominal == 0:
            sum_payouts = 0
        elif payouts > 0:
            sum_payouts = payouts 
            cur_nominal -= sum_payouts
            payout_count += 1
            if cur_nominal < 0:
                sum_payouts += cur_nominal
                cur_nominal = 0
            else:
                pass

        losses.append(sum_payouts)
    att_prob = payout_count / term
    tot_loss = np.sum(losses) /nominal
    rel_losses = np.array(losses) / nominal
    return rel_losses, att_prob, tot_loss



def init_exp_loss_att_prob_simulation(pay_dam_df, nominal, print_prob=True):
    att_prob_list = []
    annual_losses = []
    total_losses = []
    for i in range(simulated_years-term):
        events_per_year = []
        for j in range(term):
            if 'year' in pay_dam_df.columns:
                events_per_year.append(pay_dam_df[pay_dam_df['year'] == (i+j)])
            else:
                events_per_year.append(pd.DataFrame({'pay': [0], 'damage': [0]}))
        losses, att_prob, tot_loss = init_bond_exp_loss(events_per_year, nominal)

        att_prob_list.append(att_prob)
        annual_losses.extend(losses)
        total_losses.append(tot_loss)

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
    es_metrics = {'VaR_99_ann': VaR_99_ann, 'VaR_99_tot': VaR_99_tot, 'VaR_95_ann': VaR_95_ann, 'VaR_95_tot': VaR_95_tot,
                  'ES_99_ann': ES_99_ann, 'ES_99_tot': ES_99_tot, 'ES_95_ann': ES_95_ann, 'ES_95_tot': ES_95_tot}

    if print_prob:
        print(f'Expected Loss = {exp_loss_ann}')
        print(f'Attachment Probability = {att_prob}')

    return exp_loss_ann, att_prob, annual_losses, es_metrics

def init_bond_simulation(pay_dam_df, premiums, rf_rate, nominal, want_ann_returns=True, model_rf=False):

    metric_names = ['tot_payout', 'tot_damage', 'tot_pay']

    #Check if premiums/rf_rates are single values
    premiums = fct.check_scalar(premiums)

    metrics_per_premium = pd.DataFrame(index = range(len(premiums)), columns=["Premium", "Sharpe_ratio_ann", "Sharpe_ratio_tot",
                                                                              "Coverage", "Basis_risk", "Average Payments"])

    returns_per_premium = pd.DataFrame(index = range(len(premiums)), columns=["Premium","Annual", "Total"])

    for z, premium in enumerate(premiums):
        #Monte Carlo Simulation
        annual_returns = []
        tot_returns = []
        rf_annual = []
        rf_total = []
        metrics_sim = {key: [] for key in metric_names}
        for i in range(simulated_years-term):
            #model interest rates if wanted
            if model_rf:
                rf = init_model_rf(rf_rate)
            else:
                rf = rf_rate
            #create events per year of bond term
            events_per_year = []
            for j in range(term):
                if 'year' in pay_dam_df.columns:
                    events_per_year.append(pay_dam_df[pay_dam_df['year'] == (i+j)])
                else:
                    events_per_year.append(pd.DataFrame({'pay': [0], 'damage': [0]}))
            simulated_ncf_rel, metrics, rf_rates_list = init_bond(events_per_year, premium, rf, nominal)
    
            metrics_sim['tot_payout'].append(metrics['tot_payout'])
            metrics_sim['tot_damage'].append(metrics['tot_damage'])
            metrics_sim['tot_pay'].append(metrics['tot_pay'])

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
        premium_float = np.float64(premium)

        sharpe_ratio_ann = init_sharpe_ratio(annual_returns, rf_annual)
        sharpe_ratio_tot = init_sharpe_ratio(tot_returns, rf_total)

        metrics_per_premium.loc[z] = [premium_float, sharpe_ratio_ann, sharpe_ratio_tot,
                                      metrics_sim_sum['tot_payout']/metrics_sim_sum['tot_damage'], metrics_sim_sum['tot_payout']-metrics_sim_sum['tot_damage'], metrics_sim_sum['tot_pay']]  
        
        returns_per_premium.loc[z] = [premium_float, annual_returns, tot_returns]

    return metrics_per_premium, returns_per_premium

def init_bond(events_per_year, premium, risk_free_rates, nominal):
    simulated_ncf = []
    tot_payout = []
    tot_damage = []
    rf_rates_list = []
    metrics = {}    
    cur_nominal = nominal

    for k in range(term):
        rf = check_rf(risk_free_rates, k)
        rf_rates_list.append(rf)
        payouts = np.sum(events_per_year[k]['pay'].to_numpy()) 
        damages = np.sum(events_per_year[k]['damage'].to_numpy())            
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

        net_cash_flow = (cur_nominal * (premium + rf)) - sum_payouts
            

        simulated_ncf.append(net_cash_flow)
        tot_payout.append(sum_payouts)
        tot_damage.append(sum_damages)
    simulated_ncf_rel = np.array(simulated_ncf) / nominal
    metrics['tot_payout'] = np.sum(tot_payout)
    metrics['tot_damage'] = np.sum(tot_damage)
    if np.sum(tot_payout) == 0:
        tot_pay = np.nan
    else:
        tot_pay = np.sum(tot_payout)
    metrics['tot_pay'] = tot_pay

    return simulated_ncf_rel, metrics, rf_rates_list

def init_requ_premium(requ_sharpe_ratio, simulation_matrix, rf_rate, print_prem=True):

    # Define the difference function between Sharpe ratio curve and required Sharpe ratio
    def intersection_func(x):
        return np.float64(sharpe_interp(x).item()) - requ_sharpe_ratio
    
    premiums = simulation_matrix['Premium']
    sharpe_ratios = simulation_matrix['Sharpe_ratio_ann']
    # Interpolate the Sharpe ratio curve
    sharpe_interp = interp1d(premiums, sharpe_ratios, kind='linear')
    # Use fsolve to find the intersection point(s), provide a guess
    x_guess = 0.01  # Initial guess based on the range of premiums
    x_intersection = fsolve(intersection_func, x_guess)[0]
    # Calculate the corresponding Sharpe ratio at the intersection point
    y_intersection = sharpe_interp(x_intersection)
    requ_premium = x_intersection

    if print_prem:
        print(f"Intersection point using risk free interest rate of {rf_rate*100}%: Premium = {x_intersection:.4f}, Sharpe Ratio = {y_intersection:.4f}")

    return requ_premium

def display_premiums(requ_premiums, requ_sharpe_ratio, rf_rate, simulated_metrics, exp_loss):
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Premium [share of nominal]')
    ax1.set_ylabel('Sharpe ratio')
    ax1.plot(simulated_metrics['Premium'], simulated_metrics['Sharpe_ratio_ann'])

    for i in range(len(requ_sharpe_ratio)):
        risk_multiple = requ_premiums[i]/exp_loss
        color = (random.random(), random.random(), random.random())

        ax1.axhline(y = requ_sharpe_ratio[i], color = color, linestyle = '-', label='Required Sharpe Ratio') 
        print(f'Required Sharpe Ratio: {requ_sharpe_ratio[i]}; Risk free rate: {round(rf_rate*100,3)}%; Required Premium: {round(requ_premiums[i]*100,3)}%; Risk Multiple: {round(risk_multiple,3)}', )
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


def find_sharpe(premium, payout, sigma, rf, target_sharpe):
    expected_return = (premium + rf) - payout
    return (((expected_return - rf) / sigma - target_sharpe)**2)**0.5

def init_prem_sharpe_ratio(ann_losses, rf, target_sharpe):
    # Example inputs
    avg_losses = np.mean(ann_losses)
    sigma = np.std(ann_losses)

    result = minimize(lambda p: find_sharpe(p, avg_losses, sigma, rf, target_sharpe), 
                      x0=[0.05])
    optimal_premium = result.x[0]

    return optimal_premium
