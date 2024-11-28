import numpy as np
from pathlib import Path
import exposures_cc as ex_cc
import exposures as ex
import exposure_euler
import functions as fct
import impact as cimp
import bound_prot_dam as bpd
import haz_int_grd as hig
import set_nominal as snom
import alt_pay_opt as apo
import simulate_bond as sb
import prem_ibrd as prib
import simulate_multi_cty_bond as smcb
import calc_premium as cp
from colorama import Fore, Style, Back

artemis_multiplier = 4.11
ann_ret = True

def init_sng_cty_bond_principal(country, prot_share, rf_rate, target_sharpe, grid_size=600, buffer_size=1, low_to_prot=None, to_prot_share=None, int_ws=True, incl_plots=False):    
    #load tc_tracks, create hazard class and calculate exposure
    exp, applicable_basin, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex.init_TC_exp(country=country, grid_size=grid_size, buffer_size=buffer_size, load_fls=True, plot_exp=incl_plots, plot_centrs=incl_plots, plt_grd=incl_plots)
    #calculate impact and aggregate impact per grid
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=incl_plots) 
    if low_to_prot is not None: 
            imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, lower_rp=low_to_prot)
    else:
        imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, prot_share=to_prot_share, exposure=exp)
    #set up hazard intensity matrix per grid and event
    if int_ws: 
        int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_storms=tc_storms, stat=100)
    else:
        int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_tracks=storm_basin_sub)
    #set principal
    premium_dic = {}
    for ps_share in prot_share:
        ps_str = str(round(ps_share, 2))
        premium_dic[ps_str] = {'ibrd': 0, 'artemis': 0, 'regression': 0, 'required': 0, 'exp_loss': 0, 'att_prob': 0}

    premium_simulation_ps = {}
    returns_ps = {}
    pay_dam_df_ps = {}
    es_metrics_ps = {}
    ann_losses = {}

    nom_arr = []
    for i in range(len(prot_share)):
        ps_str = str(round(prot_share[i], 2))
        nominal = snom.init_nominal(impact=imp, exposure=exp, prot_share=prot_share[i])
        nom_arr.append(nominal)
        if nominal < imp_lower_rp:
            print(Back.RED + "Warning: Given Budget to small to cover specified minimal damage")
            print("The specified damage which should be covered is: ",round(imp_lower_rp, 3), " [USD]")
            print(Style.RESET_ALL)
        #optimize minimum and maximum triggering wind speed per grid cell
        result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt_flt, damages_evt=imp_per_event_flt)
        #create data frame containing payment vs damage per event
        pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_1, optimized_2, int_grid, nominal, imp_admin_evt, damages=imp_per_event)
        #calculate expected loss and attachment probability
        exp_loss_ann, att_prob, ann_losses[ps_str], es_metrics = sb.init_exp_loss_att_prob_simulation(pay_dam_df, nominal, print_prob=False)
        #calculate premiums using different approaches
        requ_prem = sb.init_prem_sharpe_ratio(ann_losses[ps_str], rf_rate, target_sharpe)
        params_ibrd = prib.init_prem_ibrd(want_plot=False)
        a, k, b = params_ibrd
        premium_dic[ps_str]['ibrd'] = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
        premium_dic[ps_str]['artemis'] = exp_loss_ann * artemis_multiplier
        premium_dic[ps_str]['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100
        premium_dic[ps_str]['required'] = requ_prem

        #simulate cat bond
        premium_simulation, returns = sb.init_bond_simulation(pay_dam_df, requ_prem, rf_rate, nominal, ann_ret) 
        premium_dic[ps_str]['exp_loss'] = exp_loss_ann
        premium_dic[ps_str]['att_prob'] = att_prob
        premium_simulation_ps[ps_str] = premium_simulation
        returns_ps[ps_str] = returns
        pay_dam_df_ps[ps_str] = pay_dam_df
        es_metrics_ps[ps_str] = es_metrics

    return premium_simulation_ps, returns_ps, premium_dic, nom_arr, pay_dam_df_ps, es_metrics_ps, int_grid, imp_per_event_flt, imp_admin_evt_flt



def init_mlt_cty_bond_principal(countries, pay_dam_df_dic_ps, prot_share, nominals_dic_ps, rf_rate, target_sharpe, int_grid_dic=None, damages_grid_flt_dic=None, damages_evt_flt_dic=None, incl_plots=False):  
    #set principal
    premium_dic = {}
    for ps_share in prot_share:
        ps_str = str(round(ps_share,2))
        premium_dic[ps_str] = {'ibrd': 0, 'artemis': 0, 'regression': 0, 'required': 0, 'exp_loss': 0, 'att_prob': 0}

    premium_simulation_ps = {}
    returns_ps = {}
    tot_coverage_prem_cty_ps = {}
    es_metrics_ps = {}
    MES_cty_ps = {}
    requ_nom_arr = []
    ann_loss_ps = {}

    l = len(prot_share)

    nominal_arr = []
    for i in range(l):
        nom_cty = []
        for cty in nominals_dic_ps.keys():
            nom_cty.append(nominals_dic_ps[cty][i])
        nominal_arr.append(np.sum(nom_cty))

    fct.print_progress_bar(0, l)

    for i in range(l):
        ps_str = str(round(prot_share[i],2))
        nominal = nominal_arr[i]

        if pay_dam_df_dic_ps is None:
            pay_dam_df_dic = {}
            for key in int_grid_dic:
                #optimize minimum and maximum triggering wind speed per grid cell
                if damages_grid_flt_dic is not None:
                    result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid_dic[key], nominal, damages_grid=damages_grid_flt_dic[key], damages_evt=None)
                    pay_dam_df = apo.alt_pay_vs_damage(damages_evt_flt_dic[key], optimized_1, optimized_2, int_grid_dic[key], nominal, damages_grid_flt_dic[key])
                else:
                    result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid_dic[key], nominal, damages_grid=None, damages_evt=damages_evt_flt_dic[key])
                    pay_dam_df = apo.alt_pay_vs_damage(damages_evt_flt_dic[key], optimized_1, optimized_2, int_grid_dic[key], nominal, damages_grid_flt_dic[key])
                pay_dam_df_dic[key] = pay_dam_df
                requ_nom = nominal
                nominal_dic_cty = None

        else:
            nominal_dic_cty = {}
            for cty in nominals_dic_ps.keys():
                nominal_dic_cty[cty] = nominals_dic_ps[cty][i]
            pay_dam_df_dic = {}
            for key, pay_dam_df in pay_dam_df_dic_ps.items():
                pay_dam_df_dic[key] = pay_dam_df[ps_str]
            exp_loss_ann, att_prob, ann_losses, total_losses, es_metrics, MES_cty = smcb.init_exp_loss_att_prob_simulation(countries, pay_dam_df_dic, nominal, nominal_dic_cty, print_prob=False)
            requ_nom = total_losses.max() * nominal

        exp_loss_ann, att_prob, ann_losses, total_losses, es_metrics, MES_cty = smcb.init_exp_loss_att_prob_simulation(countries, pay_dam_df_dic, requ_nom, nominal_dic_cty, print_prob=False)
        #calculate premiums using different approaches
        requ_prem = sb.init_prem_sharpe_ratio(ann_losses, rf_rate, target_sharpe)
        params_ibrd = prib.init_prem_ibrd(want_plot=False)
        a, k, b = params_ibrd
        premium_dic[ps_str]['ibrd'] = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
        premium_dic[ps_str]['artemis'] = exp_loss_ann * artemis_multiplier
        premium_dic[ps_str]['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100
        premium_dic[ps_str]['required'] = requ_prem
        #simulate cat bond
        premium_simulation, returns, tot_coverage_prem_cty = smcb.init_bond_simulation(pay_dam_df_dic, requ_prem, rf_rate, requ_nom, countries, nominal_dic_cty, ann_ret) 
        premium_dic[ps_str]['exp_loss'] = exp_loss_ann
        premium_dic[ps_str]['att_prob'] = att_prob
        premium_simulation_ps[ps_str] = premium_simulation
        returns_ps[ps_str] = returns
        tot_coverage_prem_cty_ps[ps_str] = tot_coverage_prem_cty
        es_metrics_ps[ps_str] = es_metrics
        MES_cty_ps[ps_str] = MES_cty
        requ_nom_arr.append(requ_nom)
        ann_loss_ps[ps_str] = ann_losses
        fct.print_progress_bar(i + 1, l)


    return premium_simulation_ps, returns_ps, tot_coverage_prem_cty_ps, premium_dic, requ_nom_arr, es_metrics_ps, MES_cty_ps, ann_loss_ps


def sng_cty_bond(country, rf_rate=0.0, target_sharpe=0.5, buffer_distance_km=105, res_exp=30, grid_size=6000, buffer_grid_size=1, prot_share=None, prot_rp=None, low_to_prot=None, to_prot_share=None, storm_dir=Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks"), output_dir=Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard"), ibrd_path=Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data"), incl_plots=False):    
    #load tc_tracks, create hazard class and calculate exposure
    exp, applicable_basin, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex.init_TC_exp(country=country, buffer_distance_km=buffer_distance_km, res_exp=res_exp, grid_size=grid_size, buffer_grid_size=buffer_grid_size, OUTPUT_DIR=output_dir, STORM_DIR=storm_dir, load_fls=True, plot_exp=incl_plots, plot_centrs=incl_plots, plt_grd=incl_plots)
    #calculate impact and aggregate impact per grid
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=incl_plots) 
    if low_to_prot is not None: 
            imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, lower_rp=low_to_prot)
    else:
        imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, prot_share=to_prot_share, exposure=exp)
    #set up hazard intensity matrix per grid and event
    int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_storms=tc_storms, stat='mean')

    premium_dic = {'ibrd': 0, 'regression': 0, 'required': 0, 'exp_loss': 0, 'att_prob': 0}


    if prot_share is not None:
        nominal = snom.init_nominal(impact=imp, exposure=exp, prot_share=prot_share)
    else:
        nominal = snom.init_nominal(impact=imp, exposure=exp, prot_rp=prot_rp)
    if nominal < imp_lower_rp:
        print(Back.RED + "Warning: Given Budget to small to cover specified minimal damage")
        print("The specified damage which should be covered is: ",round(imp_lower_rp, 3), " [USD]")
        print(Style.RESET_ALL)
    #optimize minimum and maximum triggering wind speed per grid cell
    result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt_flt, damages_evt=imp_per_event_flt, print_params=incl_plots)
    #create data frame containing payment vs damage per event
    pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_1, optimized_2, int_grid, nominal, imp_admin_evt)
    #calculate expected loss and attachment probability
    exp_loss_ann, att_prob, ann_losses, es_metrics = sb.init_exp_loss_att_prob_simulation(pay_dam_df, nominal, print_prob=False)
    #calculate premiums using different approaches
    requ_prem = sb.init_prem_sharpe_ratio(ann_losses, rf_rate, target_sharpe)
    params_ibrd = prib.init_prem_ibrd(file_path=ibrd_path,want_plot=False)
    a, k, b = params_ibrd
    ibrd_prem = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
    premium_dic['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100
    premium_dic['required'] = requ_prem
    premium_dic['ibrd'] = ibrd_prem
    premium_dic['exp_loss'] = exp_loss_ann
    premium_dic['att_prob'] = att_prob
    #simulate cat bond
    bond_metrics, bond_returns = sb.init_bond_simulation(pay_dam_df, ibrd_prem, rf_rate, nominal, ann_ret) 

    return bond_metrics, bond_returns, premium_dic, nominal, pay_dam_df, es_metrics, int_grid, imp_per_event_flt, imp_admin_evt_flt

def sng_cty_bond_cc(country, cc_model, storm_dir, output_dir, rf_rate=0.0, target_sharpe=0.5, buffer_distance_km=105, res_exp=30, grid_size=6000, buffer_grid_size=1, prot_share=None, prot_rp=None, low_to_prot=None, to_prot_share=None, ibrd_path=Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data"), incl_plots=False):    
    #load tc_tracks, create hazard class and calculate exposure
    exp, applicable_basin, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex_cc.init_TC_exp(country=country, cc_model=cc_model, file_path=output_dir, storm_path=storm_dir, buffer_distance_km=buffer_distance_km, res_exp=res_exp, grid_size=grid_size, buffer_grid_size=buffer_grid_size, load_fls=True, plot_exp=incl_plots, plot_centrs=incl_plots, plt_grd=incl_plots)
    #calculate impact and aggregate impact per grid
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=incl_plots) 
    if low_to_prot is not None: 
            imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, lower_rp=low_to_prot)
    else:
        imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, prot_share=to_prot_share, exposure=exp)
    #set up hazard intensity matrix per grid and event
    int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_storms=tc_storms, stat=90, cc_model=cc_model)

    premium_dic = {'ibrd': 0, 'regression': 0, 'required': 0, 'exp_loss': 0, 'att_prob': 0}

    if prot_share is not None:
        nominal = snom.init_nominal(impact=imp, exposure=exp, prot_share=prot_share)
    else:
        nominal = snom.init_nominal(impact=imp, exposure=exp, prot_rp=prot_rp)
    if nominal < imp_lower_rp:
        print(Back.RED + "Warning: Given Budget to small to cover specified minimal damage")
        print("The specified damage which should be covered is: ",round(imp_lower_rp, 3), " [USD]")
        print(Style.RESET_ALL)
    #optimize minimum and maximum triggering wind speed per grid cell
    result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt_flt, damages_evt=imp_per_event_flt, print_params=incl_plots)
    #create data frame containing payment vs damage per event
    pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_1, optimized_2, int_grid, nominal, imp_admin_evt)
    #calculate expected loss and attachment probability
    exp_loss_ann, att_prob, ann_losses, es_metrics = sb.init_exp_loss_att_prob_simulation(pay_dam_df, nominal, print_prob=False)
    #calculate premiums using different approaches
    requ_prem = sb.init_prem_sharpe_ratio(ann_losses, rf_rate, target_sharpe)
    params_ibrd = prib.init_prem_ibrd(file_path=ibrd_path, want_plot=False)
    a, k, b = params_ibrd
    ibrd_prem = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
    premium_dic['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100
    premium_dic['required'] = requ_prem
    premium_dic['ibrd'] = ibrd_prem
    premium_dic['exp_loss'] = exp_loss_ann
    premium_dic['att_prob'] = att_prob
    #simulate cat bond
    bond_metrics, bond_returns = sb.init_bond_simulation(pay_dam_df, ibrd_prem, rf_rate, nominal, ann_ret) 

    return bond_metrics, bond_returns, premium_dic, nominal, pay_dam_df, es_metrics, int_grid, imp_per_event_flt, imp_admin_evt_flt



def mlt_cty_bond(countries, pay_dam_df_dic, nominals_dic, rf_rate, target_sharpe, ibrd_path=Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data"), opt_cap=True, incl_plots=False):  
    #set principal
    premium_dic = {'ibrd': 0, 'regression': 0, 'required': 0, 'exp_loss': 0, 'att_prob': 0}

    nom_cty = []
    for cty in nominals_dic.keys():
        nom_cty.append(nominals_dic[cty])
    nominal = (np.sum(nom_cty))


    if opt_cap:
        exp_loss_ann, att_prob, ann_losses, total_losses, es_metrics, MES_cty = smcb.init_exp_loss_att_prob_simulation(countries, pay_dam_df_dic, nominal, nominals_dic, print_prob=False)
        requ_nom = total_losses.max() * nominal
    else:
        requ_nom = nominal
    exp_loss_ann, att_prob, ann_losses, total_losses, es_metrics, MES_cty = smcb.init_exp_loss_att_prob_simulation(countries, pay_dam_df_dic, requ_nom, nominals_dic, print_prob=False)
    #calculate premiums using different approaches
    requ_prem = sb.init_prem_sharpe_ratio(ann_losses, rf_rate, target_sharpe)
    params_ibrd = prib.init_prem_ibrd(file_path=ibrd_path, want_plot=False)
    a, k, b = params_ibrd
    premium_dic['ibrd'] = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
    premium_dic['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100
    premium_dic['required'] = requ_prem
    #simulate cat bond
    premium_simulation, returns, tot_coverage_prem_cty = smcb.init_bond_simulation(pay_dam_df_dic, requ_prem, rf_rate, requ_nom, countries, nominals_dic, ann_ret) 
    premium_dic['exp_loss'] = exp_loss_ann
    premium_dic['att_prob'] = att_prob


    return premium_simulation, returns, tot_coverage_prem_cty, premium_dic, requ_nom, es_metrics, MES_cty
