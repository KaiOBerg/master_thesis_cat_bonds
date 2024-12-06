import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
getcontext().prec = 17  
import functions as fct


term = 3
simulated_years = 10000


def init_bond_exp_loss(countries, events_per_year, nominal, nominal_dic_cty=None):
    ann_loss = []
    loss_month = pd.DataFrame(columns=['losses', 'months'])
    cur_nominal = nominal
    cur_nom_cty = nominal_dic_cty.copy() if nominal_dic_cty is not None else {int(country): 1 for country in countries}

    payout_count = 0
    cty_losses = {country: [] for country in countries}

    for k in range(term):
        losses = []
        cty_losses_event = {country: [] for country in countries}
        if events_per_year[k].empty:
            sum_payouts = [0]
            months = []
        else:
            events_per_year[k] = events_per_year[k].sort_values(by='month')
            months = events_per_year[k]['month'].tolist()
            cties = events_per_year[k]['country_code'].tolist()
            pay = events_per_year[k]['pay'].tolist()

            sum_payouts = []    
            for o in range(len(events_per_year[k])):
                payout = pay[o]
                cty = cties[o]
                if payout == 0 or cur_nominal == 0 or cur_nom_cty[int(cty)] == 0:
                    sum_payouts.append(0)
                elif payout > 0:
                    event_payout = payout
                    if nominal_dic_cty is not None:
                        cur_nom_cty[int(cty)] -= event_payout
                        if cur_nom_cty[int(cty)] < 0:
                            event_payout += cur_nom_cty[int(cty)]
                            cur_nom_cty[int(cty)] = 0
                    cur_nominal -= event_payout
                    if cur_nominal < 0:
                        event_payout += cur_nominal
                        cur_nominal = 0
                    else:
                        pass
                    sum_payouts.append(event_payout)
                    cty_losses_event[cty].append(event_payout)
            
        losses.append(np.sum(sum_payouts))
        loss_month.loc[k] = [sum_payouts, months]
        for cty in cty_losses_event.keys():
            cty_losses[cty].append(np.sum(cty_losses_event[cty]))

        ann_pay = np.sum(losses)
        if ann_pay > 0:
            payout_count += 1
        ann_loss.append(ann_pay)

    att_prob = payout_count / term
    rel_tot_loss = np.sum(ann_loss) / nominal
    rel_losses = list(np.array(ann_loss) / nominal)
    for key in cty_losses.keys():
        cty_losses[key] = [cty_abs_loss / nominal for cty_abs_loss in cty_losses[key]]
    loss_month['losses'] = loss_month['losses'].apply(lambda x: [i / nominal for i in x])


    return rel_losses, att_prob, rel_tot_loss, cty_losses, loss_month



def init_exp_loss_att_prob_simulation(countries, pay_dam_df_dic, nominal, nominal_dic_cty=None, print_prob=True):
    att_prob_list = []
    annual_losses = []
    total_losses = []
    list_loss_month = []
    ann_cty_losses = {country: [] for country in countries}

    for i in range(simulated_years-term):
        events_per_year = []
        for j in range(term):
            events_per_cty = []  
            for cty in countries:
                    events = pay_dam_df_dic[int(cty)][pay_dam_df_dic[int(cty)]['year'] == (i + j)].copy()
                    events['country_code'] = cty
                    events_per_cty.append(events)  
            year_events_df = pd.concat(events_per_cty, ignore_index=True) if events_per_cty else pd.DataFrame()
            events_per_year.append(year_events_df)

        losses, att_prob, rel_tot_loss, cty_losses,loss_month = init_bond_exp_loss(countries, events_per_year, nominal, nominal_dic_cty)

        list_loss_month.append(loss_month)
        att_prob_list.append(att_prob)
        annual_losses.extend(losses)
        total_losses.append(rel_tot_loss)

        for key in cty_losses:
            ann_cty_losses[key].extend(cty_losses[key])

    df_loss_month = pd.concat(list_loss_month, ignore_index=True)

    # Convert simulated net cash flows to a series
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
    MES_cty = {country: {'95': None, '99': None, 'EL': None} for country in ann_cty_losses.keys()}
    for country, ann_cty_losses_iter in ann_cty_losses.items():
        ann_cty_losses_iter = pd.Series(ann_cty_losses_iter)
        MES_cty[country]['95'] = ann_cty_losses_iter[annual_losses > VaR_95_ann].mean()
        MES_cty[country]['99'] = ann_cty_losses_iter[annual_losses > VaR_99_ann].mean()
    for cty in countries: 
        MES_cty[cty]['EL'] = np.mean(ann_cty_losses[cty])

    es_metrics = {'VaR_99_ann': VaR_99_ann, 'VaR_99_tot': VaR_99_tot, 'VaR_95_ann': VaR_95_ann, 'VaR_95_tot': VaR_95_tot,
                  'ES_99_ann': ES_99_ann, 'ES_99_tot': ES_99_tot, 'ES_95_ann': ES_95_ann, 'ES_95_tot': ES_95_tot}
    
    if print_prob:
        print(f'Expected Loss = {exp_loss_ann}')
        print(f'Attachment Probability = {att_prob}')

    return exp_loss_ann, att_prob, df_loss_month, total_losses, es_metrics, MES_cty

def init_bond_simulation(pay_dam_df_dic, premium, rf_rate, nominal, countries, nominal_dic_cty=None, el_dic=None, want_ann_returns=True, model_rf=False):

    nom_sng = []
    el_sum = []
    for cty in countries:
        nom_sng.append(nominal_dic_cty[cty])
        el_sum.append(el_dic[cty])
    nom_sng = sum(nom_sng)
    el_sum = sum(el_sum)
    
    share_nom = {}
    share_prem_abs = {}
    for cty in countries:
        share_nom[cty] = nominal_dic_cty[cty]/nom_sng * nominal
        share_prem_abs[cty] = (el_dic[cty]/el_sum) * (premium * nominal)

    share_prem = {}
    for cty in countries:
        share_prem[cty] = share_prem_abs[cty]/share_nom[cty]


    metric_names = ['tot_payout', 'tot_damage', 'tot_premium', 'tot_pay']

    bond_metrics = pd.DataFrame(columns=["Premium", "Sharpe_ratio_ann", "Sharpe_ratio_tot",
                                         "Coverage", "Basis_risk", "Average Payments", "Summed Payments", 'Total Premiums'])

    bond_returns = pd.DataFrame(columns=["Premium","Annual", "Total"])

    tot_coverage_cty = {}
    for code in countries:
        tot_coverage_cty[code] = {'payout': [], 'damage': [], 'coverage': [], 'count_zero': 0}

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
        events_per_year = []
        for j in range(term):
            events_per_cty = [] 
            for cty in countries:
                    events = pay_dam_df_dic[int(cty)][pay_dam_df_dic[int(cty)]['year'] == (i + j)].copy()
                    events['country_code'] = cty
                    events_per_cty.append(events)  
            year_events_df = pd.concat(events_per_cty, ignore_index=True) if events_per_cty else pd.DataFrame()
            events_per_year.append(year_events_df)

        simulated_ncf_rel, metrics, rf_rates_list, coverage_cty = init_bond(events_per_year, premium, rf, nominal, countries, nominal_dic_cty)

        metrics_sim['tot_payout'].append(metrics['tot_payout'])
        metrics_sim['tot_damage'].append(metrics['tot_damage'])
        metrics_sim['tot_premium'].append(metrics['tot_premium'])
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
    metrics_sim_sum['tot_premium'] = np.sum(metrics_sim['tot_premium'])
    metrics_sim_sum['tot_pay'] = np.nanmean(metrics_sim['tot_pay'])
    for key in tot_coverage_cty.keys():
        tot_coverage_cty[key]['coverage'] = (sum(tot_coverage_cty[key]['payout']) / sum(tot_coverage_cty[key]['damage']))
        tot_coverage_cty[key]['payout'] = (sum(tot_coverage_cty[key]['payout']))
        tot_coverage_cty[key]['damage'] = (sum(tot_coverage_cty[key]['damage']))
        tot_coverage_cty[key]['count_zero'] += tot_coverage_cty[key]['count_zero']
    premium_float = np.float64(premium)
    sharpe_ratio_ann = init_sharpe_ratio(annual_returns, rf_annual)
    sharpe_ratio_tot = init_sharpe_ratio(tot_returns, rf_total)
    bond_metrics.loc[len(bond_metrics)] = [premium_float, sharpe_ratio_ann, sharpe_ratio_tot,
                                           metrics_sim_sum['tot_payout']/metrics_sim_sum['tot_damage'], 
                                           metrics_sim_sum['tot_payout']-metrics_sim_sum['tot_damage'], 
                                           metrics_sim_sum['tot_pay'], metrics_sim_sum['tot_payout'],
                                           metrics_sim_sum['tot_premium']]  
    
    bond_returns.loc[len(bond_returns)] = [premium_float, annual_returns, tot_returns]

            
    return bond_metrics, bond_returns, tot_coverage_cty

def init_bond(events_per_year, premium, risk_free_rates, nominal, countries, nominal_dic_cty=None):   
    simulated_ncf = []
    simulated_premium = []
    tot_payout = []
    tot_damage = []
    coverage_cty = {}
    for code in countries:
        coverage_cty[code] = {'payout': 0, 'damage': 0}
    rf_rates_list = []
    metrics = {}    
    cur_nominal = nominal
    cur_nom_cty = nominal_dic_cty.copy() if nominal_dic_cty is not None else {int(country): 1 for country in countries}

        
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
            months = events_per_year[k]['month'].tolist()
            cties = events_per_year[k]['country_code'].tolist()
            pay = events_per_year[k]['pay'].tolist()
            dam = events_per_year[k]['damage'].tolist()
            ncf_pre_event = (cur_nominal * (premium + rf)) / 12 * months[0]
            net_cash_flow_ann.append(ncf_pre_event)
            premium_ann.append(cur_nominal * premium / 12 * (months[0]))
            cty_payouts_event = {country: [] for country in countries}
            cty_damages_event = {country: [] for country in countries}
            for o in range(len(events_per_year[k])):
                payout = pay[o]
                cty = cties[o]
                damage = dam[o]
                month = months[o]

                if payout == 0 or cur_nominal == 0 or cur_nom_cty[int(cty)] == 0:
                    event_payout = 0
                elif payout > 0:
                    event_payout = payout
                    if nominal_dic_cty is not None:
                        cur_nom_cty[int(cty)] -= event_payout
                        if cur_nom_cty[int(cty)] < 0:
                            event_payout += cur_nom_cty[int(cty)]
                            cur_nom_cty[int(cty)] = 0
                    cur_nominal -= event_payout
                    if cur_nominal < 0:
                        event_payout += cur_nominal
                        cur_nominal = 0
                    else:
                        pass
                if o + 1 < len(events_per_year[k]):
                    nex_month = months[o+1] 
                    premium_post_event = (cur_nominal * premium) / 12 * (nex_month - month)
                    ncf_post_event = ((cur_nominal * (premium + rf)) / 12 * (nex_month - month)) - event_payout
                else:
                    premium_post_event = (cur_nominal * premium) / 12 * (12- month)
                    ncf_post_event = ((cur_nominal * (premium + rf)) / 12 * (12- month)) - event_payout

                net_cash_flow_ann.append(ncf_post_event)
                premium_ann.append(premium_post_event)
                sum_payouts_ann.append(event_payout)
                sum_damages_ann.append(damage)
                cty_payouts_event[cty].append(event_payout)
                cty_damages_event[cty].append(damage)

            for key in cty_payouts_event.keys():
                coverage_cty[key]['payout'] += sum(cty_payouts_event[key])
                coverage_cty[key]['damage'] += sum(cty_damages_event[key])

        tot_payout.append(np.sum(sum_payouts_ann))
        tot_damage.append(np.sum(sum_damages_ann))
        simulated_ncf.append(np.sum(net_cash_flow_ann))
        simulated_premium.append(np.sum(premium_ann))
    simulated_ncf_rel = list(np.array(simulated_ncf) / nominal)
    metrics['tot_payout'] = np.sum(tot_payout)
    metrics['tot_damage'] = np.sum(tot_damage)
    metrics['tot_premium'] = np.sum(simulated_premium)
    if np.sum(tot_payout) == 0:
        tot_pay = np.nan
    else:
        tot_pay = np.sum(tot_payout)
    metrics['tot_pay'] = tot_pay

    return simulated_ncf_rel, metrics, rf_rates_list, coverage_cty

def init_sharpe_ratio(rel_returns, risk_free_rate, exp_short=None):
    exp_ret = np.mean(rel_returns)
    rf = np.mean(risk_free_rate)
    if exp_short: 
        std = exp_short
    else:
        std = np.std(rel_returns)
    sharpe_ratio = (exp_ret - rf) / std
    return sharpe_ratio

def init_model_rf(risk_free_rate):

    modeled_rf_rates = [risk_free_rate] * term

    return modeled_rf_rates

def check_rf(risk_free_rates, iterator):

    if isinstance(risk_free_rates, list):
        rf = risk_free_rates[iterator]
    else:
        rf = risk_free_rates

    return rf


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

def init_prem_sharpe_ratio(ann_losses, rf, target_sharpe):        

    result = minimize(lambda p: find_sharpe(p, ann_losses, rf, target_sharpe), 
                      x0=[0.05])
    optimal_premium = result.x[0]

    return optimal_premium