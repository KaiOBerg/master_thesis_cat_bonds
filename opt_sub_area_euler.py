import impact as cimp
import bound_prot_dam as bpd
import alt_pay_opt as apo
import set_nominal as snom
import haz_int_grd as hig
import exposures_alt as aexp

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

OUTPUT_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard") #Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks") #Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")

#choose country
country = 882
#define bond charactersitcs
prot_rp = 250
lower_share = 0.05
int_stat = np.arange(10, 101, 10)
selected_buffer = 105

def sng_cty_bond(country, grid_specs, int_stat, prot_rp, file_path, storm_path, buffer_distance, buffer_grid, to_prot_share, incl_plots=False):    
    #load tc_tracks, create hazard class and calculate exposured
    exp, applicable_basin, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = aexp.init_TC_exp(country=country, grid_specs=grid_specs, file_path=file_path, storm_path=storm_path, buffer_grid_size=buffer_grid,
                                                                                              buffer_distance_km=buffer_distance, load_fls=True, res_exp=30, plot_exp=incl_plots, plot_centrs=incl_plots, plt_grd=incl_plots)
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=incl_plots) 
    imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, prot_share=to_prot_share, exposure=exp)
    
    nominal = snom.init_nominal(impact=imp, exposure=exp, prot_rp=prot_rp, print_nom=False)

    basis_risk_dic = {}
    for i in int_stat:
        int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_storms=tc_storms, stat=float(i))
        result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt_flt, damages_evt=imp_per_event_flt, print_params=incl_plots)
        pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_1, optimized_2, int_grid, nominal, imp_admin_evt)
        pay_dam_df['damage'] = pay_dam_df['damage'].apply(lambda value: min(value, nominal))
        #basis_risk_dic[i] = np.sum(pay_dam_df['damage']) - np.sum(pay_dam_df['pay'])
        basis_risk_dic[i] = np.sqrt(mean_squared_error(pay_dam_df['damage'], pay_dam_df['pay']))
 


    int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_storms=tc_storms, stat='mean')
    result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt_flt, damages_evt=imp_per_event_flt, print_params=incl_plots)
    pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_1, optimized_2, int_grid, nominal, imp_admin_evt)
    pay_dam_df['damage'] = pay_dam_df['damage'].apply(lambda value: min(value, nominal))
    #basis_risk_dic['mean'] = np.sum(pay_dam_df['damage']) - np.sum(pay_dam_df['pay'])
    basis_risk_dic['mean'] = np.sqrt(mean_squared_error(pay_dam_df['damage'], pay_dam_df['pay']))

    basis_risk_df = pd.DataFrame([basis_risk_dic])

    return basis_risk_df, len(optimized_1), exp

all_dfs = []

br, y, exp = sng_cty_bond(country, [1,1], int_stat, prot_rp=prot_rp, to_prot_share=lower_share, buffer_distance=selected_buffer, buffer_grid=20, file_path=OUTPUT_DIR, storm_path=STORM_DIR, incl_plots=False)
br['Count grids'] = y
all_dfs.append(br)

for i in range(10):

    grid_specs = [1+i, 1+i]
    br, y, exp = sng_cty_bond(country, grid_specs, int_stat, prot_rp=prot_rp, to_prot_share=lower_share, buffer_distance=selected_buffer, buffer_grid=1, file_path=OUTPUT_DIR, storm_path=STORM_DIR, incl_plots=False)
    br['Count grids'] = y
    all_dfs.append(br)

combined_br = pd.concat(all_dfs, ignore_index=True)
combined_br.to_excel(OUTPUT_DIR / f"rmse_grids_{country}.xlsx", index=False)
