import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import functions as fct


term = 3
num_simulations = 20000


def init_bond_exp_loss(pay_dam_df, nominal, event_probability):
    payouts = pay_dam_df['pay'].to_numpy()
    losses = []
    cur_nominal = nominal
    payout_count = 0

    for i in range(term):
        #randomly generate number of events in one year using poisson distribution and calculated yearly event probability
        num_events = np.random.poisson(lam=event_probability)
        #If there are events in the year, sample that many payouts and the associated damages
        if num_events == 0 or cur_nominal == 0:
            sum_payouts = 0
        elif num_events > 0:
            random_indices = np.random.randint(0, len(payouts), size=num_events)
            sum_payouts = np.sum(payouts[random_indices]) 
            if sum_payouts > 0:
                cur_nominal += sum_payouts
                payout_count += 1
                if cur_nominal < 0:
                    sum_payouts += cur_nominal
                    cur_nominal = 0
                else:
                    pass
            

        losses.append(sum_payouts)
    att_prob = payout_count / term
    losses = np.mean(losses)
    rel_losses = losses / nominal
    return losses, rel_losses, att_prob



def init_exp_loss_att_prob_simulation(pay_dam_df, nominal, event_probability):
    att_prob_list = []
    #Monte Carlo Simulation
    annual_losses = []
    annual_losses_rel = []
    for _ in range(num_simulations):
        losses, rel_losses, att_prob = init_bond_exp_loss(pay_dam_df, nominal, event_probability)

        att_prob_list.append(att_prob)
        annual_losses.append(losses)
        annual_losses_rel.append(rel_losses)

    # Convert simulated net cash flows to a series
    att_prob = np.mean(att_prob_list)
    annual_losses = pd.Series(annual_losses)
    annual_losses_rel = pd.Series(annual_losses_rel)
    #calculate finacial metrics
    exp_loss_ann = annual_losses_rel.mean()
            

    return exp_loss_ann, att_prob

def init_bond_simulation(pay_dam_df, premiums, rf_rates, event_probability, nominal, want_ann_returns=True, model_rf=False):

    metric_names = ['att_prob', 'coverage', 'basis_risk', 'tot_pay']

    #Check if premiums/rf_rates are single values
    premiums = fct.check_scalar(premiums)
    rf_rates = fct.check_scalar(rf_rates)

    returns_rf = {}
    metrics_rf = {}

    for rf_iter in rf_rates:
        rf_str = str(rf_iter)
        #model interest rates when wanted
        if model_rf:
            rf_iter = init_model_rf(rf_iter)
        else:
            pass

        metrics_per_premium = pd.DataFrame(index = range(len(premiums)), columns=["Premium", "Sharpe_ratio_ann","VaR_01_ann", "ES_01_ann", "Attachment_probability_ann", "Annual_expected_loss",
                                                                              "Sharpe_ratio_tot","VaR_01_tot", "ES_01_tot", "Attachment_probability_bond", "Bond_expected_loss",
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
                simulated_ncf_rel, metrics, rf_rates_list = init_bond(pay_dam_df, premium, rf_iter, nominal, event_probability)
        
                metrics_sim['att_prob'].append(metrics['att_prob'])
                metrics_sim['coverage'].append(metrics['coverage'])
                metrics_sim['basis_risk'].append(metrics['basis_risk'])
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
            metrics_sim_mean = {key: np.nanmean(values_sim) for key, values_sim in metrics_sim.items()}
            VaR_01_ann = annual_returns.quantile(0.01)
            VaR_01_tot = tot_returns.quantile(0.01)
            ES_01_ann = annual_returns[annual_returns < VaR_01_ann].mean()
            ES_01_tot = tot_returns[tot_returns < VaR_01_tot].mean()
            exp_loss_ann = init_expected_loss(annual_returns)
            exp_loss_tot = init_expected_loss(tot_returns)
            att_prob_tot =  sum(1 for value in metrics_sim['tot_pay'] if value > 0) / num_simulations
            premium_float = np.float64(premium)
    
            sharpe_ratio_ann = init_sharpe_ratio(annual_returns, rf_annual)
            sharpe_ratio_tot = init_sharpe_ratio(tot_returns, rf_total)
    
            metrics_per_premium.loc[i] = [premium_float, sharpe_ratio_ann, VaR_01_ann, ES_01_ann, metrics_sim_mean['att_prob'], exp_loss_ann,
                                          sharpe_ratio_tot, VaR_01_tot, ES_01_tot, att_prob_tot, exp_loss_tot, 
                                          metrics_sim_mean['coverage'], metrics_sim_mean['basis_risk'], metrics_sim_mean['tot_pay']]  
            
            returns_per_premium.loc[i] = [premium_float, annual_returns, tot_returns]

            
        returns_rf[rf_str] = returns_per_premium
        metrics_rf[rf_str] = metrics_per_premium

    return metrics_rf, returns_rf

def init_bond(pay_dam_df, premium, risk_free_rates, nominal, event_probability):
    payouts = pay_dam_df['pay'].to_numpy()
    damages = pay_dam_df['damage'].to_numpy()
    simulated_ncf = []
    tot_payout = []
    tot_damage = []
    rf_rates_list = []
    payout_count = 0
    metrics = {}    
    cur_nominal = nominal
    payout_happened = False

    for i in range(term):
        rf = check_rf(risk_free_rates, i)
        rf_rates_list.append(rf)
        net_cash_flow = 0
        #randomly generate number of events in one year using poisson distribution and calculated yearly event probability
        num_events = np.random.poisson(lam=event_probability)
        #If there are events in the year, sample that many payouts and the associated damages
        if num_events == 0 and not payout_happened or cur_nominal == 0:
            net_cash_flow = cur_nominal * (premium + rf)
            sum_damages = 0
            sum_payouts = 0
        elif num_events > 0:
            random_indices = np.random.randint(0, len(payouts), size=num_events)
            sum_payouts = np.sum(payouts[random_indices])
            sum_damages = np.sum(damages[random_indices])
            if sum_payouts == 0 and not payout_happened:
                net_cash_flow = cur_nominal * (premium + rf)
            elif sum_payouts > 0:
                net_cash_flow = 0 - sum_payouts
                cur_nominal += net_cash_flow
                payout_count += 1
                payout_happened = True
                if cur_nominal < 0:
                    net_cash_flow = (cur_nominal - net_cash_flow) * -1
                    sum_payouts = sum_payouts - cur_nominal
                    cur_nominal = 0
                else:
                    pass
            

        simulated_ncf.append(net_cash_flow)
        tot_payout.append(sum_payouts)
        tot_damage.append(sum_damages)
    simulated_ncf_rel = simulated_ncf / nominal
    coverage, basis_risk, tot_pay = init_coverage(tot_payout, tot_damage)
    metrics['coverage'] = coverage
    metrics['basis_risk'] = basis_risk
    metrics['att_prob'] = payout_count / term
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

def init_sharpe_ratio(rel_returns, risk_free_rate):
    std = np.std(rel_returns)
    exp_ret = np.mean(rel_returns)
    rf = np.mean(risk_free_rate)
    sharpe_ratio = (exp_ret - rf) / std
    return sharpe_ratio

def init_coverage(payouts, damages):
    if np.sum(payouts) == np.sum(damages) == 0:
        coverage = np.nan
        basis_risk = np.nan
        tot_pay = np.nan
    elif np.sum(payouts) == 0: 
        coverage = 0
        basis_risk = -np.sum(damages)
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