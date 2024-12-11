import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
getcontext().prec = 17  
import functions as fct
import time

term = 3
simulated_years = 10000


def init_bond_exp_loss(countries, events_per_year, nominal, nominal_dic_cty=None):
    ann_loss = np.zeros(term)  
    loss_month_data = []
    cur_nominal = nominal
    cur_nom_cty = nominal_dic_cty.copy() if nominal_dic_cty is not None else {int(country): 1 for country in countries}

    payout_count = 0
    cty_losses = {country: np.zeros(term) for country in countries}  

    for k in range(term):
        losses = np.zeros(1)  
        cty_losses_event = {country: [] for country in countries}
        sum_payouts = np.zeros(len(events_per_year[k]))

        if not events_per_year[k].empty:
            events = events_per_year[k].sort_values(by='month')
            months = events['month'].to_numpy()
            cties = events['country_code'].to_numpy()
            pay = events['pay'].to_numpy()

            sum_payouts = np.zeros(len(events))  
            
            for o in range(len(events)):
                payout = pay[o]
                cty = cties[o]
                
                if payout == 0 or cur_nominal == 0 or cur_nom_cty[int(cty)] == 0:
                    event_payout = 0
                else:
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

                sum_payouts[o] = event_payout
                cty_losses_event[cty].append(event_payout)

            losses[0] = np.sum(sum_payouts)
            for cty, cty_loss in cty_losses_event.items():
                cty_losses[cty][k] = np.sum(cty_loss)
        else:
            losses[0] = 0
            months = []
        
        ann_loss[k] = losses[0]
        loss_month_data.append((sum_payouts, months))
        if losses[0] > 0:
            payout_count += 1

    loss_month = pd.DataFrame(loss_month_data, columns=['losses', 'months'])

    att_prob = payout_count / term
    rel_tot_loss = np.sum(ann_loss) / nominal
    rel_losses = list(np.array(ann_loss) / nominal)
    for key in cty_losses.keys():
        cty_losses[key] = cty_losses[key] / nominal 
    loss_month['losses'] = loss_month['losses'].values / nominal
    return rel_losses, att_prob, rel_tot_loss, cty_losses, loss_month



def init_exp_loss_att_prob_simulation(countries, pay_dam_df_dic, nominal, nominal_dic_cty, confidence_levels=[0.95, 0.99], print_prob=True):
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

        losses, att_prob, rel_tot_loss, cty_losses, loss_month = init_bond_exp_loss(countries, events_per_year, nominal, nominal_dic_cty)

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

    risk_metrics_annual = multi_level_es(annual_losses, confidence_levels)
    risk_metrics_total = multi_level_es(total_losses, confidence_levels)

    #MES_cty = {country: {'EL': None} for country in ann_cty_losses.keys()} # MES_cty = {country: {'95': None, '99': None, 'EL': None} for country in ann_cty_losses.keys()}
    #for country, ann_cty_losses_iter in ann_cty_losses.items():
    #    ann_cty_losses_iter = pd.Series(ann_cty_losses_iter)
    #    MES_cty[country]['95'] = ann_cty_losses_iter[annual_losses > VaR_95_ann].mean()
    #    MES_cty[country]['99'] = ann_cty_losses_iter[annual_losses > VaR_99_ann].mean()
    MES_cty = {country: {'EL': np.mean(ann_cty_losses[country])} for country in ann_cty_losses}
    for cty in MES_cty:
        MES_cty[cty]['share_EL'] = MES_cty[cty]['EL'] / exp_loss_ann

    es_metrics = {'VaR_99_ann': risk_metrics_annual[0.99]['VaR'], 'VaR_99_tot': risk_metrics_total[0.99]['VaR'], 'VaR_95_ann': risk_metrics_annual[0.95]['VaR'], 'VaR_95_tot': risk_metrics_total[0.95]['VaR'],
                  'ES_99_ann': risk_metrics_annual[0.99]['ES'], 'ES_99_tot': risk_metrics_total[0.99]['ES'], 'ES_95_ann': risk_metrics_annual[0.95]['ES'], 'ES_95_tot': risk_metrics_total[0.95]['ES']}
    
    if print_prob:
        print(f'Expected Loss = {exp_loss_ann}')
        print(f'Attachment Probability = {att_prob}')
    return exp_loss_ann, att_prob, df_loss_month, total_losses, es_metrics, MES_cty

def init_bond_simulation(pay_dam_df_dic, premium, rf_rate, nominal, countries, nominal_dic_cty=None, el_dic=None, want_ann_returns=True, model_rf=False):

    metric_names = ['tot_payout', 'tot_damage', 'tot_premium', 'tot_pay']

    bond_metrics = pd.DataFrame(columns=["Premium", "Sharpe_ratio_ann", "Sharpe_ratio_tot",
                                         "Coverage", "Basis_risk", "Average Payments", "Summed Payments", 'Total Premiums'])

    bond_returns = pd.DataFrame(columns=["Premium","Annual", "Total"])

    tot_coverage_cty = {}
    for cty in countries:
        tot_coverage_cty[cty] = {'payout': [], 'damage': [], 'coverage': [], 'count_zero': 0}

    for cty in countries:
        annual_returns = {cty: [] for cty in countries}
    annual_returns['Total'] = []
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
            for key in countries:
                annual_returns[key].extend(simulated_ncf_rel[key])
            annual_returns['Total'].extend(simulated_ncf_rel['Total'])
        else:
            ann_return = (1 + sum(simulated_ncf_rel)) ** (1/term) - 1
            annual_returns.append(ann_return)
        tot_returns.append(np.sum(simulated_ncf_rel['Total']))
        rf_annual.append(np.mean(rf_rates_list))
        rf_total.append(np.sum(rf_rates_list))
    # Convert simulated net cash flows to a series
    for key in annual_returns:
        annual_returns[key] = pd.Series(annual_returns[key])
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

def init_bond(events_per_year, premium, risk_free_rates, nominal, countries, nominal_dic_cty=None, share_prem=None, nom_prem=None):   
    simulated_ncf = []
    simulated_premium = []
    simulated_ncf_cty = {cty: [] for cty in countries}
    simulated_premium_cty = {cty: [] for cty in countries}
    simulated_ncf_rel_cty = {}
    tot_payout = []
    tot_damage = []
    coverage_cty = {}
    for code in countries:
        coverage_cty[code] = {'payout': 0, 'damage': 0}
    rf_rates_list = []
    metrics = {}    
    cur_nominal = nominal
    cur_nom_cty = nominal_dic_cty.copy() if nominal_dic_cty is not None else {int(country): 1 for country in countries}
    cur_nom_cty_prem = nom_prem.copy() 
        
    for k in range(term):
        rf = check_rf(risk_free_rates, k)
        rf_rates_list.append(rf)
        premium_ann = 0
        net_cash_flow_ann = 0
        premium_ann_cty = {code: 0 for code in countries}
        net_cash_flow_ann_cty = {code: 0 for code in countries}
        if events_per_year[k].empty:
            for code in countries: 
                premium_ann_cty[code] += cur_nom_cty_prem[code]  * share_prem[code]
                net_cash_flow_ann_cty[code] += (cur_nom_cty_prem[code] * (share_prem[code] + rf))
            premium_ann += cur_nominal  * premium
            net_cash_flow_ann += (cur_nominal * (premium + rf))
            sum_payouts_ann = 0
            sum_damages_ann = 0
        else:
            events_per_year[k] = events_per_year[k].sort_values(by='month')
            sum_payouts_ann = []
            sum_damages_ann = []
            months = events_per_year[k]['month'].tolist()
            cties = events_per_year[k]['country_code'].tolist()
            pay = events_per_year[k]['pay'].tolist()
            dam = events_per_year[k]['damage'].tolist()
            for code in countries: 
                premium_ann_cty[code] += cur_nom_cty_prem[code]  * share_prem[code]  / 12 * months[0]
                net_cash_flow_ann_cty[code] += (cur_nom_cty_prem[code] * (share_prem[code] + rf)) / 12 * months[0]
            premium_ann += cur_nominal  * premium  / 12 * months[0]
            net_cash_flow_ann += (cur_nominal * (premium + rf)) / 12 * months[0]
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
                        cur_nom_cty_prem[int(cty)] -= event_payout
                        if cur_nom_cty[int(cty)] < 0:
                            event_payout += cur_nom_cty[int(cty)]
                            cur_nom_cty[int(cty)] = 0
                        if cur_nom_cty_prem[cty] < 0: 
                            cur_nom_cty_prem[cty] = 0
                    cur_nominal -= event_payout
                    if cur_nominal < 0:
                        event_payout += cur_nominal
                        cur_nominal = 0
                    else:
                        pass
                if o + 1 < len(events_per_year[k]):
                    nex_month = months[o+1] 
                    for code in countries: 
                        premium_ann_cty[code] += cur_nom_cty_prem[code] * share_prem[code]  / 12 * (nex_month - month)
                        ncf_pre = (cur_nom_cty_prem[code] * (share_prem[code] + rf)) / 12 * (nex_month - month) - event_payout
                        net_cash_flow_ann_cty[code] += (cur_nom_cty_prem[code] * (share_prem[code] + rf)) / 12 * (nex_month - month) - event_payout
                    premium_ann += cur_nominal * premium  / 12 * (nex_month - month)
                    net_cash_flow_ann += (cur_nominal * (premium + rf)) / 12 * (nex_month - month) - event_payout
                else:
                    for code in countries: 
                        premium_ann_cty[code] += cur_nom_cty_prem[code]  * share_prem[code]  / 12 * (12- month)
                        ncf_pre = (cur_nom_cty_prem[code] * (share_prem[code] + rf)) / 12 * (12- month) - event_payout
                        if ncf_pre < -1:
                            net_cash_flow_ann_cty[code] = -1
                        net_cash_flow_ann_cty[code] += (cur_nom_cty_prem[code] * (share_prem[code] + rf)) / 12 * (12- month) - event_payout
                    premium_ann += cur_nominal  * premium  / 12 * (12- month)
                    net_cash_flow_ann += (cur_nominal * (premium + rf)) / 12 * (12- month) - event_payout

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
        for code in countries:
            simulated_ncf_cty[code].append(np.sum(net_cash_flow_ann_cty[code]))
            simulated_premium_cty[code].append(np.sum(premium_ann_cty[code]))
    simulated_ncf_rel = list(np.array(simulated_ncf) / nominal)
    for code in countries:
            simulated_ncf_rel_cty[code] = list(np.array(simulated_ncf_cty[code]) / nom_prem[code])
    metrics['tot_payout'] = np.sum(tot_payout)
    metrics['tot_damage'] = np.sum(tot_damage)
    metrics['tot_premium'] = np.sum(simulated_premium)
    if np.sum(tot_payout) == 0:
        tot_pay = np.nan
    else:
        tot_pay = np.sum(tot_payout)
    metrics['tot_pay'] = tot_pay

    NCF = simulated_ncf_rel_cty.copy()
    NCF['Total'] = simulated_ncf_rel

    return NCF, metrics, rf_rates_list, coverage_cty

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


def multi_level_es(losses, confidence_levels):
    """
    Calculate VaR and ES for multiple confidence levels.
    
    Parameters:
    - losses: array-like, list of losses
    - confidence_levels: list of floats, confidence levels (e.g., [0.95, 0.99])
    
    Returns:
    - risk_metrics: dict, VaR and ES values keyed by confidence level
    """
    # Convert losses to a NumPy array
    losses = np.array(losses)
    
    # Sort losses once
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    
    risk_metrics = {}
    for cl in confidence_levels:
        # Calculate index for VaR
        var_index = int(np.ceil(n * cl)) - 1
        var = sorted_losses[var_index]
        
        # Calculate ES
        tail_losses = sorted_losses[var_index + 1:]
        es = tail_losses.mean() if len(tail_losses) > 0 else var
        
        # Store metrics
        risk_metrics[cl] = {'VaR': var, 'ES': es}
    
    return risk_metrics


def simulate_ncf_prem(premium, ann_losses, tranches, cty_el_dic, rf=0.0):
    ncf = {tranche['RP']: [] for _, tranche in tranches.iterrows()}
    premiums_tot = []
    premiums_tot_alt = []
    ncf_tot = []
    cur_nominal = 1
    for i in range(len(ann_losses)):
        losses = ann_losses['losses'].iloc[i]
        months = ann_losses['months'].iloc[i]
        if np.sum(losses) == 0:
            premiums_tot.append(cur_nominal * premium)
            ncf_tmp = cur_nominal * (premium + rf)
            ncf_tot.append(ncf_tmp)
            prem_it_alt = 0
            for _, tranche in tranches.iterrows():
                ncf[tranche['RP']].append(cur_nominal * tranche['nominal'] * (tranche['premium'] + rf))
                prem_it_alt += cur_nominal * tranche['nominal'] * tranche['premium']
            premiums_tot_alt.append(prem_it_alt)
        else:
            ncf_tot_tmp = []
            ncf_tmp = {tranche['RP']: [] for _, tranche in tranches.iterrows()}
            premiums_tot_tmp = []
            premiums_tot_tmp.append(cur_nominal * premium / 12 * months[0])
            prem_pre_tmp = cur_nominal * (premium + rf) / 12 * months[0]
            ncf_tot_tmp.append(prem_pre_tmp)
            prem_it_alt = 0
            premiums_tot_tmp_alt = []
            for _, tranche in tranches.iterrows():
                ncf_tmp[tranche['RP']].append(cur_nominal * tranche['nominal'] * (tranche['premium'] + rf) / 12 * months[0])
                prem_it_alt += cur_nominal * tranche['nominal'] * tranche['premium'] / 12 * months[0]
            premiums_tot_tmp_alt.append(prem_it_alt)
            for j in range(len(losses)):
                loss = losses[j]
                month = months[j]
                cur_nominal -= loss
                if cur_nominal < 0:
                    loss += cur_nominal
                    cur_nominal = 0
                if j + 1 < len(losses):
                    nex_month = months[j+1]
                    premiums_tot_tmp.append(cur_nominal * premium / 12 * (nex_month - month))
                    prem_tmp = ((cur_nominal * (premium + rf)) / 12 * (nex_month - month))
                    ncf_tot_tmp.append(prem_tmp - loss)
                    prem_it_alt = 0
                    for _, tranche in tranches.iloc[::-1].iterrows():
                        ncf_tmp[tranche['RP']].append(((cur_nominal * tranche['nominal'] * (tranche['premium'] + rf)) / 12 * (nex_month - month)))
                        prem_it_alt += cur_nominal * tranche['nominal'] * tranche['premium'] / 12 * (nex_month - month)
                    premiums_tot_tmp_alt.append(prem_it_alt)
                else:
                    premiums_tot_tmp.append(cur_nominal * premium / 12 * (12 - month))
                    prem_tmp = ((cur_nominal * (premium + rf)) / 12 * (12- month))
                    ncf_tot_tmp.append(prem_tmp - loss)
                    prem_it_alt = 0
                    for _, tranche in tranches.iloc[::-1].iterrows():
                        ncf_tmp[tranche['RP']].append(((cur_nominal * tranche['nominal'] * (tranche['premium'] + rf)) / 12 * (12- month)))
                        prem_it_alt += cur_nominal * tranche['nominal'] * tranche['premium'] / 12 * (12- month)
                    premiums_tot_tmp_alt.append(prem_it_alt)
            ncf_tot.append(np.sum(ncf_tot_tmp))
            tmp_loss = np.sum(losses)
            for _, tranche in tranches.iloc[::-1].iterrows():
                to_cover = tmp_loss - tranche['lower_bound']
                if to_cover < 0:
                    to_cover = 0
                ncf[tranche['RP']].append(np.sum(ncf_tmp[tranche['RP']]) - to_cover)
                tmp_loss -= to_cover
            premiums_tot.append(np.sum(premiums_tot_tmp))
            premiums_tot_alt.append(np.sum(premiums_tot_tmp_alt))
        if (i + 1) % term == 0:
            cur_nominal = 1

    ncf['Total'] = ncf_tot
    prem_cty_dic = {country: [] for country in cty_el_dic}
    for country in prem_cty_dic:
        prem_cty_dic[country] = np.array(premiums_tot) * cty_el_dic[country]['share_EL']
    prem_cty_dic['Total'] = premiums_tot
    prem_cty_dic['Total_alt'] = premiums_tot_alt
    return ncf, prem_cty_dic


def init_equ_nom_sim(events_per_year, nominal_dic_cty):
    ann_loss = np.zeros(term)
    cur_nom_cty = nominal_dic_cty.copy()

    for k in range(term):
        if not events_per_year[k].empty:
            events = events_per_year[k]
            payouts = events['pay'].to_numpy()
            cties = events['country_code'].to_numpy()

            sum_payouts = np.zeros(len(events))

            for idx, (payout, cty) in enumerate(zip(payouts, cties)):
                if payout == 0 or cur_nom_cty[cty] == 0:
                    event_payout = 0
                else:
                    event_payout = payout
                    cur_nom_cty[cty] -= event_payout
                    if cur_nom_cty[cty] < 0:
                        event_payout += cur_nom_cty[cty]
                        cur_nom_cty[cty] = 0
                sum_payouts[idx] = event_payout

            ann_loss[k] = np.sum(sum_payouts)
        else:
            ann_loss[k] = 0

    tot_loss = np.sum(ann_loss)
    return tot_loss



def requ_nom(countries, pay_dam_df_dic, nominal_dic_cty):
    total_losses = []

    for i in range(simulated_years-term):
        events_per_year = []
        for j in range(term):
            events_per_cty = [pay_dam_df_dic[int(cty)].loc[pay_dam_df_dic[int(cty)]['year'] == (i + j)].assign(country_code=cty) for cty in countries]

            year_events_df = pd.concat(events_per_cty, ignore_index=True) if events_per_cty else pd.DataFrame()
            events_per_year.append(year_events_df)

        tot_loss = init_equ_nom_sim(events_per_year, nominal_dic_cty)

        total_losses.append(tot_loss)

    requ_nominal = np.max(total_losses)

    return requ_nominal