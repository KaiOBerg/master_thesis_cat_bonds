import exposures as ex
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
params_ibrd = prib.init_prem_ibrd(want_plot=False)
a, k, b = params_ibrd
ann_ret = True

def init_sng_cty_bond(country, prot_share, low_to_prot, rf_rate, target_sharpe, int_ws=True, incl_plots=False):    
    #load tc_tracks, create hazard class and calculate exposure
    exp, applicable_basin, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex.init_TC_exp(country=country, load_fls=True, plot_exp=incl_plots, plot_centrs=incl_plots, plt_grd=incl_plots)
    #calculate impact and aggregate impact per grid
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=incl_plots) 
    imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, low_to_prot)
    #set up hazard intensity matrix per grid and event
    if int_ws: 
        int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_storms=tc_storms, stat=100)
    else:
        int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_tracks=storm_basin_sub)
    #set principal
    premium_dic = {}
    for ps_share in prot_share:
        ps_str = str(ps_share)
        premium_dic[ps_str] = {'ibrd': 0, 'artemis': 0, 'regression': 0, 'required': 0, 'exp_loss': 0, 'att_prob': 0}

    premium_simulation_ps = {}
    returns_ps = {}
    pay_dam_df_ps = {}
    es_metrics_ps = {}

    nom_arr = []
    for i in range(len(prot_share)):
        ps_str = str(prot_share[i])
        nominal, tot_exp, nom_rel_exp = snom.init_nominal(impact=imp, exposure=exp, prot_share=prot_share[i])
        nom_arr.append(nominal)
        if nominal < imp_lower_rp:
            print(imp_lower_rp)
            print(Back.RED + "Warning: Given Budget to small to cover specified minimal damage")
            print(Style.RESET_ALL)
        #optimize minimum and maximum triggering wind speed per grid cell
        result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt_flt, damages_evt=imp_per_event_flt)
        #create data frame containing payment vs damage per event
        pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_1, optimized_2, int_grid, nominal, damages=imp_per_event)
        #calculate expected loss and attachment probability
        exp_loss_ann, att_prob, ann_losses, es_metrics = sb.init_exp_loss_att_prob_simulation(pay_dam_df, nominal)
        #calculate premiums using different approaches
        requ_prem = sb.init_prem_sharpe_ratio(ann_losses, rf_rate, target_sharpe)
        premium_dic[ps_str]['ibrd'] = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
        premium_dic[ps_str]['artemis'] = exp_loss_ann * artemis_multiplier
        premium_dic[ps_str]['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100
        premium_dic[ps_str]['required'] = requ_prem

        #simulate cat bond
        premium_simulation, returns = sb.init_bond_simulation(pay_dam_df, requ_prem, rf_rate, nominal, ann_ret) #simulate cat bond using a Monte Carlo simulation
        premium_dic[ps_str]['exp_loss'] = exp_loss_ann
        premium_dic[ps_str]['att_prob'] = att_prob
        premium_simulation_ps[ps_str] = premium_simulation
        returns_ps[ps_str] = returns
        pay_dam_df_ps[ps_str] = pay_dam_df
        es_metrics_ps[ps_str] = es_metrics

    return premium_simulation_ps, returns_ps, premium_dic, nom_arr, pay_dam_df_ps, es_metrics_ps, int_grid, imp_per_event_flt, imp_admin_evt_flt



def init_mlt_cty_bond(countries, pay_dam_df_dic_ps, prot_share, nominals_lst_ps, rf_rate, target_sharpe, int_grid_dic=None, damages_grid_flt_dic=None, damages_evt_flt_dic=None, incl_plots=False):    
    #set principal
    premium_dic = {}
    for ps_share in prot_share:
        ps_str = str(ps_share)
        premium_dic[ps_str] = {'ibrd': 0, 'artemis': 0, 'regression': 0, 'required': 0, 'exp_loss': 0, 'att_prob': 0}

    premium_simulation_ps = {}
    returns_ps = {}
    tot_coverage_prem_cty_ps = {}
    es_metrics_ps = {}
    MES_cty_ps = {}
    requ_nom_arr = []


    nominal_arr = [a + b for a, b in zip(nominals_lst_ps[0], nominals_lst_ps[1])]


    for i in range(len(prot_share)):
        ps_str = str(prot_share[i])
        nominal = nominal_arr[i]
        nominal_lst_cty = []
        for k in range(len(nominals_lst_ps)):
            nominal_lst_cty.append(nominals_lst_ps[k][i])

        if pay_dam_df_dic_ps is None:
            pay_dam_df_dic = {}
            for key in int_grid_dic:
                #optimize minimum and maximum triggering wind speed per grid cell
                result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid_dic[key], nominal, damages_grid=damages_grid_flt_dic[key], damages_evt=damages_evt_flt_dic[key])
                if damages_grid_flt_dic is not None:
                    pay_dam_df = apo.alt_pay_vs_damage(damages_grid_flt_dic[key], optimized_1, optimized_2, int_grid_dic[key], nominal, include_plot=incl_plots)
                else:
                    pay_dam_df = apo.alt_pay_vs_damage(damages_evt_flt_dic[key], optimized_1, optimized_2, int_grid_dic[key], nominal, include_plot=incl_plots)
                pay_dam_df_dic[key] = pay_dam_df
                requ_nom = nominal

        else:
            #calculate expected loss and attachment probability
            pay_dam_df_dic = {}
            for key, pay_dam_df in pay_dam_df_dic_ps.items():
                pay_dam_df_dic[key] = pay_dam_df[ps_str]
            exp_loss_ann, att_prob, ann_losses, total_losses, es_metrics, MES_cty = smcb.init_exp_loss_att_prob_simulation(pay_dam_df_dic, nominal, nominal_lst_cty)
            requ_nom = total_losses.max() * nominal
            print(requ_nom)

        exp_loss_ann, att_prob, ann_losses, total_losses, es_metrics, MES_cty = smcb.init_exp_loss_att_prob_simulation(pay_dam_df_dic, requ_nom)
        #calculate premiums using different approaches
        requ_prem = sb.init_prem_sharpe_ratio(ann_losses, rf_rate, target_sharpe)
        premium_dic[ps_str]['ibrd'] = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
        premium_dic[ps_str]['artemis'] = exp_loss_ann * artemis_multiplier
        premium_dic[ps_str]['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100
        premium_dic[ps_str]['required'] = requ_prem
        #simulate cat bond
        premium_simulation, returns, tot_coverage_prem_cty = smcb.init_bond_simulation(pay_dam_df_dic, requ_prem, rf_rate, requ_nom, countries, ann_ret) #simulate cat bond using a Monte Carlo simulation
        premium_dic[ps_str]['exp_loss'] = exp_loss_ann
        premium_dic[ps_str]['att_prob'] = att_prob
        premium_simulation_ps[ps_str] = premium_simulation
        returns_ps[ps_str] = returns
        tot_coverage_prem_cty_ps[ps_str] = tot_coverage_prem_cty
        es_metrics_ps[ps_str] = es_metrics
        MES_cty_ps[ps_str] = MES_cty
        requ_nom_arr.append(requ_nom)


    return premium_simulation_ps, returns_ps, tot_coverage_prem_cty_ps, premium_dic, requ_nom_arr, es_metrics_ps, MES_cty_ps