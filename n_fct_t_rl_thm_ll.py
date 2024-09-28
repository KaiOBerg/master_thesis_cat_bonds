import exposures as ex
import impact as cimp
import bound_prot_dam as bpd
import haz_int_grd as hig
import set_nominal as snom
import alt_pay_opt as apo
import simulate_bond as sb
import prem_ibrd as prib
import calc_premium as cp

artemis_multiplier = 4.11
params_ibrd = prib.init_prem_ibrd(want_plot=False)
a, k, b = params_ibrd
ann_ret = True

def init_sng_cty_bond(country, prot_share, low_to_prot, rf_rate, target_sharpe, int_ws=True, incl_plots=False):    
    #load tc_tracks, create hazard class and calculate exposure
    exp, applicable_basin, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex.init_TC_exp(country=country, load_fls=True, plot_exp=incl_plots, plot_centrs=incl_plots, plt_grd=incl_plots)
    #calculate impact and aggregate impact per grid
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=incl_plots) 
    imp_per_event_flt = bpd.init_imp_flt(imp_per_event, low_to_prot)
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
        #optimize minimum and maximum triggering wind speed per grid cell
        result, optimized_step1, optimized_step2, optimized_step3 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt, damages_evt=imp_per_event_flt)
        #create data frame containing payment vs damage per event
        pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_step1, optimized_step2, optimized_step3, int_grid, nominal, include_plot=incl_plots)
        #calculate expected loss and attachment probability
        exp_loss_ann, att_prob, ann_losses, es_metrics = sb.init_exp_loss_att_prob_simulation(pay_dam_df, nominal)
        #calculate premiums using different approaches
        requ_prem = sb.init_prem_sharpe_ratio(ann_losses, rf_rate, target_sharpe)
        premium_dic[ps_str]['ibrd'] = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
        premium_dic[ps_str]['artemis'] = exp_loss_ann * artemis_multiplier
        premium_dic[ps_str]['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100
        premium_dic[ps_str]['required'] = requ_prem


        #print(f'The premium based on past IBRD bonds is {round(premium_ibrd*100, 3)}%')
        #print(f'The premium based on the artemis multiplier is {round(premium_artemis*100, 3)}%')
        #print(f'The premium based on the regression model from Chatoro et al. 2022 is {round(premium_regression*100, 3)}%')

        #simulate cat bond
        premium_simulation, returns = sb.init_bond_simulation(pay_dam_df, requ_prem, rf_rate, nominal, ann_ret) #simulate cat bond using a Monte Carlo simulation
        premium_dic[ps_str]['exp_loss'] = exp_loss_ann
        premium_dic[ps_str]['att_prob'] = att_prob
        premium_simulation_ps[ps_str] = premium_simulation
        returns_ps[ps_str] = returns
        pay_dam_df_ps[ps_str] = pay_dam_df
        es_metrics_ps[ps_str] = es_metrics

    return premium_simulation_ps, returns_ps, premium_dic, nom_arr, pay_dam_df_ps, es_metrics_ps, int_grid, imp_per_event_flt