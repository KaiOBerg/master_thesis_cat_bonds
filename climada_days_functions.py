import math
import matplotlib.pyplot as plt
import numpy as np
import calc_premium as cp
import prem_ibrd as prib
import alt_pay_opt as apo
import functions as fct
import simulate_bond as sb

def sng_bond_nom(arr_nominal, int_grid, imp_admin_evt_flt, imp_per_event_flt, rf_rate, target_sharpe):
    optimized_xs_nom = {}
    optimized_ys_nom = {}
    pay_dam_df_nom = {}
    bond_metrics = {}
    returns = {}
    ann_losses = {}
    es_metrics = {}
    premium_dic = {'ibrd': {}, 'regression': {}, 'required': {}, 'exp_loss': {}, 'att_prob': {}}

    l = len(arr_nominal)
    i = 0

    fct.print_progress_bar(0, l)

    for nom in arr_nominal:
        i+= 1
        nom_str = str(round(nom,0))
        result_nom, optimized_xs_nom[nom_str], optimized_ys_nom[nom_str] = apo.init_alt_optimization(int_grid, nom, damages_grid=imp_admin_evt_flt, print_params=False)

        pay_dam_df_nom[nom_str] = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_xs_nom[nom_str], optimized_ys_nom[nom_str], int_grid, nom, imp_admin_evt_flt)

        exp_loss_ann, att_prob, ann_losses[nom_str], es_metrics[nom_str] = sb.init_exp_loss_att_prob_simulation(pay_dam_df_nom[nom_str], nom, print_prob=False)
        params_ibrd = prib.init_prem_ibrd(want_plot=False)
        a, k, b = params_ibrd
        premium_ibrd = prib.monoExp(exp_loss_ann * 100, a, k, b) * exp_loss_ann
        premium_regression = cp.calc_premium_regression(exp_loss_ann * 100) / 100
        requ_prem = sb.init_prem_sharpe_ratio(ann_losses[nom_str], rf_rate, target_sharpe)
        premium_dic['ibrd'][nom_str] = premium_ibrd
        premium_dic['regression'][nom_str] = premium_regression
        premium_dic['required'][nom_str] = requ_prem
        premium_dic['exp_loss'][nom_str] = exp_loss_ann
        premium_dic['att_prob'][nom_str] = att_prob

        bond_metrics[nom_str], returns[nom_str] = sb.init_bond_simulation(pay_dam_df_nom[nom_str], premium_ibrd, rf_rate, nom)

        fct.print_progress_bar(i, l)

    coverage_nom = []
    basis_risk_nom = []

    for nom_it in arr_nominal:
        nom_str = str(round(nom_it,0))
        coverage_nom.append(bond_metrics[nom_str]['Coverage'])
        basis_risk_nom.append((bond_metrics[nom_str]['Basis_risk'])*-1)

    fig, ax1 = plt.subplots(figsize=(6, 4))

    color = 'tab:red'
    ax1.plot(arr_nominal, coverage_nom, color=color)
    ax1.set_xlabel('Principal [USD]')
    ax1.set_ylabel('Coverage []', color=color)
    ax1.set_ylim(0.3,1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Basis Risk [USD]', color=color)  # we already handled the x-label with ax1
    ax2.plot(arr_nominal, basis_risk_nom, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    premium_ibrd_arr = np.array(fct.get_all_values(premium_dic['ibrd']))
    premium_regression_arr = np.array(fct.get_all_values(premium_dic['regression']))
    premium_required_arr = np.array(fct.get_all_values(premium_dic['required']))


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 4))

    #ax1.plot(prot_share_arr, premium_artemis.values(), label='Artemis')
    ax1.plot(arr_nominal, premium_ibrd_arr, label='IBRD')
    ax1.plot(arr_nominal, premium_regression_arr, label='Regression')
    ax1.plot(arr_nominal, premium_required_arr, label='Sharpe Ratio = 0.5')
    ax1.set_xlabel('Principal [USD]')
    ax1.set_ylabel('Premium [share of principal]')
    ax1.legend(loc='upper right')

    ax2.plot(arr_nominal, (premium_ibrd_arr * arr_nominal), label='IBRD')
    ax2.plot(arr_nominal, (premium_regression_arr * arr_nominal), label='Regression')
    ax2.plot(arr_nominal, (premium_required_arr * arr_nominal), label='Sharpe Ratio = 0.5')
    ax2.set_xlabel('Principal [USDP]')
    ax2.set_ylabel('Premium [USD]')
    ax2.legend(loc='upper left')

    plt.show()

    return bond_metrics, returns, premium_dic, es_metrics, pay_dam_df_nom, optimized_xs_nom, optimized_ys_nom