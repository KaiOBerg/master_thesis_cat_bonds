'''Script for simulating single-country CAT bonds'''
import pandas as pd
import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from decimal import getcontext
getcontext().prec = 17  

term = 3
simulated_years = 1000

'''Simulate one term of bond to derive losses'''
def init_bond_exp_loss(events_per_year, nominal):
    losses = []
    loss_month = pd.DataFrame(columns=['losses', 'months'])
    cur_nominal = nominal
    payout_count = 0

    for k in range(term):

        if events_per_year[k].empty:
            sum_payouts = [0]
            months = []
        else:
            events_per_year[k] = events_per_year[k].sort_values(by='month')
            pay_tot = np.sum(events_per_year[k]['pay'].to_numpy())
            months = events_per_year[k]['month'].tolist()
            if pay_tot > 0 and cur_nominal != 0:
                payout_count += 1

            sum_payouts = []
            for o in range(len(events_per_year[k])):
                payout = events_per_year[k].loc[events_per_year[k].index[o], 'pay']
                #If there are events in the year, sample that many payouts and the associated damages
                if payout == 0 or cur_nominal == 0:
                    sum_payouts.append(0)
                elif payout > 0:
                    event_payout = payout 
                    cur_nominal -= event_payout
                    if cur_nominal < 0:
                        event_payout += cur_nominal
                        cur_nominal = 0
                    else:
                        pass
                    sum_payouts.append(event_payout)

        losses.append(np.sum(sum_payouts))
        loss_month.loc[k] = [sum_payouts, months]
    att_prob = payout_count / term
    tot_loss = np.sum(losses) /nominal
    rel_losses = np.array(losses) / nominal
    loss_month['losses'] = loss_month['losses'].apply(lambda x: [i / nominal for i in x])
    return rel_losses, att_prob, tot_loss, loss_month


'''Loop over all terms of bond to derive losses'''
def init_exp_loss_att_prob_simulation(pay_dam_df, nominal, print_prob=True):
    att_prob_list = []
    annual_losses = []
    total_losses = []
    list_loss_month = []
    for i in range(simulated_years-term):
        events_per_year = []
        for j in range(term):
            if 'year' in pay_dam_df.columns:
                events_per_year.append(pay_dam_df[pay_dam_df['year'] == (i+j)])
            else:
                events_per_year.append(pd.DataFrame({'pay': [0], 'damage': [0]}))
        losses, att_prob, tot_loss, loss_month = init_bond_exp_loss(events_per_year, nominal)
        list_loss_month.append(loss_month)

        att_prob_list.append(att_prob)
        annual_losses.extend(losses)
        total_losses.append(tot_loss)
    
    df_loss_month = pd.concat(list_loss_month, ignore_index=True)

    att_prob = np.mean(att_prob_list)
    exp_loss_ann = np.mean(annual_losses)

    annual_losses = pd.Series(annual_losses)
    total_losses = pd.Series(total_losses)

    VaR_99_ann = annual_losses.quantile(0.99)
    VaR_99_tot = total_losses.quantile(0.99)
    VaR_95_ann = annual_losses.quantile(0.95)
    VaR_95_tot = total_losses.quantile(0.95)
    if VaR_99_ann == 1:
        ES_99_ann = 1
    else:
        ES_99_ann = annual_losses[annual_losses > VaR_99_ann].mean()
    if VaR_99_tot == 1:
        ES_99_tot = 1
    else:
        ES_99_tot = total_losses[total_losses > VaR_99_tot].mean()
    if VaR_95_ann == 1:
        ES_95_ann = 1
    else:
        ES_95_ann = annual_losses[annual_losses > VaR_95_ann].mean()
    if VaR_95_tot == 1:
        ES_95_tot = 1
    else:
        ES_95_tot = total_losses[total_losses > VaR_95_tot].mean()
    es_metrics = {'VaR_99_ann': VaR_99_ann, 'VaR_99_tot': VaR_99_tot, 'VaR_95_ann': VaR_95_ann, 'VaR_95_tot': VaR_95_tot,
                  'ES_99_ann': ES_99_ann, 'ES_99_tot': ES_99_tot, 'ES_95_ann': ES_95_ann, 'ES_95_tot': ES_95_tot}

    if print_prob:
        print(f'Expected Loss = {exp_loss_ann}')
        print(f'Attachment Probability = {att_prob}')

    return exp_loss_ann, att_prob, df_loss_month, es_metrics


'''Simulate over all terms of bond to derive returns'''
def init_bond_simulation(pay_dam_df, premium, rf_rate, nominal, want_ann_returns=True, model_rf=False):

    metric_names = ['tot_payout', 'tot_damage', 'tot_premium', 'tot_pay']

    #Check if premiums/rf_rates are single values

    bond_metrics = pd.DataFrame(columns=["Premium", "Sharpe_ratio_ann", "Sharpe_ratio_tot",
                                         "Coverage", "Basis_risk", "Average Payments", "Summed Payments", 'Total Premiums'])

    bond_returns = pd.DataFrame(columns=["Premium","Annual", "Total"])

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
        metrics_sim['tot_premium'].append(metrics['tot_premium'])
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
    metrics_sim_sum['tot_premium'] = np.sum(metrics_sim['tot_premium'])
    metrics_sim_sum['tot_pay'] = np.nanmean(metrics_sim['tot_pay'])
    premium_float = np.float64(premium)
    sharpe_ratio_ann = init_sharpe_ratio(annual_returns, rf_annual)
    sharpe_ratio_tot = init_sharpe_ratio(tot_returns, rf_total)
    bond_metrics.loc[len(bond_metrics)] = [premium_float, sharpe_ratio_ann, sharpe_ratio_tot,
                                           metrics_sim_sum['tot_payout']/metrics_sim_sum['tot_damage'], 
                                           metrics_sim_sum['tot_payout']-metrics_sim_sum['tot_damage'], 
                                           metrics_sim_sum['tot_pay'], metrics_sim_sum['tot_payout'],
                                           metrics_sim_sum['tot_premium']]
    
    bond_returns.loc[len(bond_returns)] = [premium_float, annual_returns, tot_returns]

    return bond_metrics, bond_returns

'''Simulate over one term of bond to derive returns'''
def init_bond(events_per_year, premium, risk_free_rates, nominal):
    simulated_ncf = []
    simulated_premiums = []
    tot_payout = []
    tot_damage = []
    rf_rates_list = []
    metrics = {}    
    cur_nominal = nominal
    for k in range(term):
        rf = check_rf(risk_free_rates, k)
        rf_rates_list.append(rf)
        if events_per_year[k].empty:
            premium_ann = cur_nominal * premium
            net_cash_flow_ann = (cur_nominal * (premium + rf))
            sum_payouts_ann = 0
            sum_damages_ann = 0
        else:
            events_per_year[k] = events_per_year[k].sort_values(by='month')
            net_cash_flow_ann = []
            premium_ann = []
            sum_payouts_ann = []
            sum_damages_ann = []
            month = events_per_year[k].loc[events_per_year[k].index[0], 'month'] 
            ncf_pre_event = (cur_nominal * (premium + rf)) / 12 * month
            net_cash_flow_ann.append(ncf_pre_event)
            premium_ann.append((cur_nominal * premium) / 12 * month)
            for o in range(len(events_per_year[k])):
                payouts = events_per_year[k].loc[events_per_year[k].index[o], 'pay']
                damages = events_per_year[k].loc[events_per_year[k].index[o], 'damage']
                month = events_per_year[k].loc[events_per_year[k].index[o], 'month']    
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
                if o + 1 < len(events_per_year[k]):
                    nex_month = events_per_year[k].loc[events_per_year[k].index[o + 1], 'month'] 
                    premium_post_event = (cur_nominal * premium) / 12 * (nex_month - month)
                    ncf_post_event = ((cur_nominal * (premium + rf)) / 12 * (nex_month - month)) - sum_payouts
                else:
                    premium_post_event = (cur_nominal * premium) / 12 * (12- month)
                    ncf_post_event = ((cur_nominal * (premium + rf)) / 12 * (12- month)) - sum_payouts

                net_cash_flow_ann.append(ncf_post_event)
                premium_ann.append(premium_post_event)
                sum_payouts_ann.append(sum_payouts)
                sum_damages_ann.append(damages)

        simulated_ncf.append(np.sum(net_cash_flow_ann))
        simulated_premiums.append(np.sum(premium_ann))
        tot_payout.append(np.sum(sum_payouts_ann))
        tot_damage.append(np.sum(sum_damages_ann))
    simulated_ncf_rel = np.array(simulated_ncf) / nominal
    metrics['tot_payout'] = np.sum(tot_payout)
    metrics['tot_damage'] = np.sum(tot_damage)
    metrics['tot_premium'] = np.sum(simulated_premiums)
    if np.sum(tot_payout) == 0:
        tot_pay = np.nan
    else:
        tot_pay = np.sum(tot_payout)
    metrics['tot_pay'] = tot_pay

    return simulated_ncf_rel, metrics, rf_rates_list

'''plot function to see relationship between premiums and sharpe ratio -> not used for final results'''
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

'''Calculate Sharpe ratio'''
def init_sharpe_ratio(rel_returns, risk_free_rate, exp_short=None):
    exp_ret = np.mean(rel_returns)
    rf = np.mean(risk_free_rate)
    if exp_short: 
        std = exp_short
    else:
        std = np.std(rel_returns)
    sharpe_ratio = (exp_ret - rf) / std
    return sharpe_ratio

'''calculate coverare, basis risk, and total payments'''
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

'''Calculate expected loss based on returns -> not used for final results'''
def init_expected_loss(returns):
    #Filter to get the returns that are negative -> losses
    losses = returns.apply(lambda x: 0 if x > 0 else x)
    loss_magnitudes = -losses
    expected_loss = np.mean(loss_magnitudes)
    return expected_loss

'''Can be adjusted to model risk free rate but idea was discarded for now -> not used for final results'''
def init_model_rf(risk_free_rate):

    modeled_rf_rates = [risk_free_rate] * term

    return modeled_rf_rates

'''Check if risk free rate is am array or not'''
def check_rf(risk_free_rates, iterator):

    if isinstance(risk_free_rates, list):
        rf = risk_free_rates[iterator]
    else:
        rf = risk_free_rates

    return rf

'''Benchmark pricing function -> goes through all losses and determines requuired premium for ceratin sharpe ratio'''
def find_sharpe(premium, ann_losses, rf, target_sharpe):
    ncf = []
    cur_nominal = 1
    for i in range(len(ann_losses)):
        losses = ann_losses['losses'].iloc[i]
        months = ann_losses['months'].iloc[i]
        if np.sum(losses) == 0:
            ncf.append(cur_nominal * (premium + rf))
        else:
            ncf_pre_event = (cur_nominal * (premium + rf)) / 12 * (months[0])
            ncf_post_event = []
            for j in range(len(losses)):
                loss = losses[j]
                month = months[j]
                cur_nominal -= loss
                if cur_nominal < 0:
                    loss += cur_nominal
                    cur_nominal = 0
                if j + 1 < len(losses):
                    nex_month = months[j+1]
                    ncf_post_event.append(((cur_nominal * (premium + rf)) / 12 * (nex_month - month)) - loss)
                else:
                    ncf_post_event.append(((cur_nominal * (premium + rf)) / 12 * (12- month)) - loss)
            ncf.append(ncf_pre_event + np.sum(ncf_post_event))
        if (i + 1) % term == 0:
            cur_nominal = 1

    avg_ret = np.mean(ncf)
    sigma = np.std(ncf)
    return ((avg_ret - rf) / sigma - target_sharpe)**2

'''Benchmark pricing function'''
def init_prem_sharpe_ratio(ann_losses, rf, target_sharpe):        

    result = minimize(lambda p: find_sharpe(p, ann_losses, rf, target_sharpe), 
                      x0=[0.05])
    optimal_premium = result.x[0]

    return optimal_premium
