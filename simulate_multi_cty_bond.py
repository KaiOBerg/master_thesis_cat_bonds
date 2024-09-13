import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import functions as fct


term = 3
num_simulations = 20000


def init_bond_exp_loss(pay_dam_df_dic, nominal, event_probabilities):
    payouts = []
    for key in pay_dam_df_dic.keys():
        payouts.append(pay_dam_df_dic[key]['pay'].to_numpy())    
    ann_loss = []
    cur_nominal = nominal
    payout_count = 0

    for i in range(term):
        #randomly generate number of events in one year using poisson distribution and calculated yearly event probability
        losses = []
        for j in range(len(event_probabilities)):
            num_events = np.random.poisson(lam=event_probabilities[j])
            #If there are events in the year, sample that many payouts and the associated damages
            if num_events == 0 or cur_nominal == 0:
                sum_payouts = 0
            elif num_events > 0:
                random_indices = np.random.randint(0, len(payouts[j]), size=num_events)
                sum_payouts = np.sum(payouts[j][random_indices]) 
                if sum_payouts > 0:
                    cur_nominal -= sum_payouts
                    if cur_nominal < 0:
                        sum_payouts += cur_nominal
                        cur_nominal = 0
                    else:
                        pass
            

            losses.append(sum_payouts)
        ann_pay = np.sum(losses)
        if ann_pay > 0:
            payout_count += 1
        ann_loss.append(ann_pay)
    att_prob = payout_count / term
    tot_loss = np.mean(ann_loss)
    rel_losses = tot_loss / nominal
    return rel_losses, att_prob



def init_exp_loss_att_prob_simulation(pay_dam_df, nominal, event_probabilities, print_prob=True):
    att_prob_list = []
    #Monte Carlo Simulation
    annual_losses = []
    for _ in range(num_simulations):
        losses, att_prob = init_bond_exp_loss(pay_dam_df, nominal, event_probabilities)

        att_prob_list.append(att_prob)
        annual_losses.append(losses)

    # Convert simulated net cash flows to a series
    att_prob = np.mean(att_prob_list)
    exp_loss_ann = np.mean(annual_losses)
    if print_prob:
        print(f'Expected Loss = {exp_loss_ann}')
        print(f'Attachment Probability = {att_prob}')

    return exp_loss_ann, att_prob

def init_bond_simulation(pay_dam_df_dic, premiums, rf_rate, event_probabilities, nominal, want_ann_returns=True, model_rf=False):

    metric_names = ['tot_payout', 'tot_damage', 'tot_pay']

    #Check if premiums/rf_rates are single values
    premiums = fct.check_scalar(premiums)

    metrics_per_premium = pd.DataFrame(index = range(len(premiums)), columns=["Premium", "Sharpe_ratio_ann", "Cond_sharpe_ratio_01", "Cond_sharpe_ratio_05", "VaR_01_ann", "ES_01_ann",
                                                                              "Sharpe_ratio_tot","VaR_01_tot", "ES_01_tot",
                                                                              "Coverage", "Basis_risk", "Average Payments"])

    returns_per_premium = pd.DataFrame(index = range(len(premiums)), columns=["Premium","Annual", "Total"])
    for i, premium in enumerate(premiums):
        #Monte Carlo Simulation
        annual_returns = []
        tot_returns = []
        rf_annual = []
        rf_total = []
        metrics_sim = {key: [] for key in metric_names}
        for _ in range(num_simulations):
            #model interest rates if wanted
            if model_rf:
                rf = init_model_rf(rf_rate)
            else:
                rf = rf_rate
            simulated_ncf_rel, metrics, rf_rates_list = init_bond(pay_dam_df_dic, premium, rf, nominal, event_probabilities, model_rf)
    
            metrics_sim['tot_payout'].append(metrics['tot_payout'])
            metrics_sim['tot_damage'].append(metrics['tot_damage'])
            metrics_sim['tot_pay'].append(metrics['tot_pay'])

            if want_ann_returns:
                ann_return = np.mean(simulated_ncf_rel)
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
        VaR_01_ann = annual_returns.quantile(0.01)
        VaR_01_tot = tot_returns.quantile(0.01)
        VaR_05_ann = annual_returns.quantile(0.05)
        VaR_05_tot = tot_returns.quantile(0.05)
        ES_01_ann = annual_returns[annual_returns < VaR_01_ann].mean()
        ES_01_tot = tot_returns[tot_returns < VaR_01_tot].mean()
        ES_05_ann = annual_returns[annual_returns < VaR_05_ann].mean()
        ES_05_tot = tot_returns[tot_returns < VaR_05_tot].mean()
        premium_float = np.float64(premium)

        sharpe_ratio_ann = init_sharpe_ratio(annual_returns, rf_annual)
        sharpe_ratio_tot = init_sharpe_ratio(tot_returns, rf_total)
        cond_sharpe_ratio_01 = init_sharpe_ratio(annual_returns, rf_total, ES_01_ann)
        cond_sharpe_ratio_05 = init_sharpe_ratio(annual_returns, rf_total, ES_05_ann)

        metrics_per_premium.loc[i] = [premium_float, sharpe_ratio_ann, cond_sharpe_ratio_01, cond_sharpe_ratio_05, VaR_01_ann, ES_01_ann,
                                      sharpe_ratio_tot, VaR_01_tot, ES_01_tot, 
                                      metrics_sim_sum['tot_payout']/metrics_sim_sum['tot_damage'], metrics_sim_sum['tot_payout']-metrics_sim_sum['tot_damage'], metrics_sim_sum['tot_pay']]  
        
        returns_per_premium.loc[i] = [premium_float, annual_returns, tot_returns]

            
    return metrics_per_premium, returns_per_premium

def init_bond(pay_dam_df_dic, premium, risk_free_rates, nominal, event_probabilities, model_rf=False):
    payouts = []
    damages = []
    for key in pay_dam_df_dic.keys():
        payouts.append(pay_dam_df_dic[key]['pay'].to_numpy())    
        damages.append(pay_dam_df_dic[key]['damage'].to_numpy())    
    simulated_ncf = []
    tot_payout = []
    tot_damage = []
    rf_rates_list = []
    metrics = {}    
    cur_nominal = nominal

    for i in range(term):
        rf = check_rf(risk_free_rates, i)
        rf_rates_list.append(rf)
        net_cash_flow = 0
        country_payouts = []
        country_damages = []
        country_ncf = []
        for j in range(len(event_probabilities)):
            #randomly generate number of events in one year using poisson distribution and calculated yearly event probability
            num_events = np.random.poisson(lam=event_probabilities[j])
            #If there are events in the year, sample that many payouts and the associated damages
            if num_events == 0:
                sum_damages = 0
                sum_payouts = 0
            elif num_events > 0:
                random_indices = np.random.randint(0, len(payouts[j]), size=num_events)
                sum_payouts = np.sum(payouts[j][random_indices])
                sum_damages = np.sum(damages[j][random_indices])
                if sum_payouts > 0:
                    cur_nominal -= sum_payouts
                    if cur_nominal < 0:
                        sum_payouts = sum_payouts + cur_nominal
                        cur_nominal = 0
                    else:
                        pass
            
            country_payouts.append(sum_payouts)
            country_damages.append(sum_damages)
            country_ncf.append(net_cash_flow)

        ann_pay = np.sum(country_payouts)

        if ann_pay == 0:
            net_cash_flow = cur_nominal * (premium + rf)
        else: 
            net_cash_flow = (cur_nominal * (premium + rf)) - ann_pay

        tot_payout.append(ann_pay)
        simulated_ncf.append(np.sum(net_cash_flow))
        tot_damage.append(np.sum(country_damages))
    simulated_ncf_rel = simulated_ncf / nominal
    metrics['tot_payout'] = np.sum(tot_payout)
    metrics['tot_damage'] = np.sum(tot_damage)
    if np.sum(tot_payout) == 0:
        tot_pay = np.nan
    else:
        tot_pay = np.sum(tot_payout)
    metrics['tot_pay'] = tot_pay

    return simulated_ncf_rel, metrics, rf_rates_list

def init_requ_premium(requ_sharpe_ratio, simulation_matrix, rf_rates):

    # Define the difference function between Sharpe ratio curve and required Sharpe ratio
    def intersection_func(x):
        return np.float64(sharpe_interp(x).item()) - requ_sharpe_ratio
    
    requ_premiums = {}

    rf_rates = fct.check_scalar(rf_rates)

    for rf in rf_rates:
        rf_str = str(rf)
        premiums = simulation_matrix[rf_str]['Premium']
        sharpe_ratios = simulation_matrix[rf_str]['Sharpe_ratio_tot']

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
        plt.xlabel('Premium')
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