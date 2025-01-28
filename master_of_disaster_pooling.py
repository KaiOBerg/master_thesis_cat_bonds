'''Create single-country CAT bonds for all countries in the pool'''
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import n_fct_t_rl_thm_ll as bond_fct


#define directories 
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
IBRD_DIR = Path("/cluster/work/climate/kbergmueller")

#choose country
countries = [480, 212, 882, 332, 670, 28, 388, 52, 662, 659, 308, 214, 44, 548, 242, 780, 192, 570, 84, 776, 90, 174, 184, 584, 585]
countries_150 = [332, 388, 214, 44, 548, 192, 84, 90] 
fiji = [242]
countries_30 = [480, 212, 670, 28, 52, 662, 659, 308, 882, 780, 570, 776, 174, 184, 584, 585]

countries_str = ['MUS', 'DMA', 'WSM', 'HTI', 'VCT', 'ATG', 'JAM', 'BRB', 'LCA', 'KNA', 'GRD', 
                 'DOM', 'BHS', 'VUT', 'FJI', 'TTO', 'CUB', 'NIU', 'BLZ', 'TON', 'SLB', 'COM', 'COK', 'MHL', 'PLW']
countries_str_x_label = ['MUS', 'DMA', 'WSM', 'HTI', 'VCT', 'ATG', 'JAM', 'BRB', 'LCA', 'KNA', 'GRD', 
                         'DOM', 'BHS', 'VUT', 'FJI', 'TTO', 'CUB', 'NIU', 'BLZ', 'TON', 'SLB', 'COM', 'COK', 'MHL', 'PLW', 'Pool']

#set risk free rate
rf_rates = 0.00
#set sharpe ratio to beat
target_sharpe = 0.5
#define bond setting
lower_share = 0.05
prot_rp = 250

#define empty dictionaries
bond_metrics_sng_dic = {}
returns_sng_dic = {}
premium_dic_sng_dic = {}
nominal_sng_dic = {}
pay_dam_df_sng_dic = {}
es_metrics_sng_dic = {}
int_grid_sng_dic = {}
imp_per_event_flt_sng_dic = {}
imp_admin_evt_flt_sng_dic = {}
ann_losses_dic = {}


for cty in countries:
    if cty in bond_metrics_sng_dic:
        print(f"Bond for {cty} already calculated, skipping.")
        continue
    print(f'Create bond for {cty}')
    if cty in countries_150:
        bond_metrics, returns, premium_dic, nominal, pay_dam_df, es_metrics, int_grid, imp_per_event_flt, imp_admin_evt_flt, ann_losses = bond_fct.sng_cty_bond(country=cty,
                                                                                                                                                                prot_rp=prot_rp, 
                                                                                                                                                                to_prot_share=lower_share,
                                                                                                                                                                buffer_distance_km=105,
                                                                                                                                                                res_exp=150,
                                                                                                                                                                grid_size=1000,
                                                                                                                                                                grid_specs=[5,5],
                                                                                                                                                                buffer_grid_size=5,
                                                                                                                                                                incl_plots=False,
                                                                                                                                                                plt_save=True,
                                                                                                                                                                storm_dir=STORM_DIR,
                                                                                                                                                                output_dir=OUTPUT_DIR,
                                                                                                                                                                ibrd_path=IBRD_DIR)
    if cty in countries_30:
        bond_metrics, returns, premium_dic, nominal, pay_dam_df, es_metrics, int_grid, imp_per_event_flt, imp_admin_evt_flt, ann_losses = bond_fct.sng_cty_bond(country=cty,
                                                                                                                                                                prot_rp=prot_rp, 
                                                                                                                                                                to_prot_share=lower_share,
                                                                                                                                                                buffer_distance_km=105,
                                                                                                                                                                res_exp=30,
                                                                                                                                                                grid_size=1000,
                                                                                                                                                                grid_specs=[3,3],
                                                                                                                                                                buffer_grid_size=1,
                                                                                                                                                                incl_plots=False,
                                                                                                                                                                plt_save=True,
                                                                                                                                                                storm_dir=STORM_DIR,
                                                                                                                                                                output_dir=OUTPUT_DIR,
                                                                                                                                                                ibrd_path=IBRD_DIR)  
    if cty in fiji:
        bond_metrics, returns, premium_dic, nominal, pay_dam_df, es_metrics, int_grid, imp_per_event_flt, imp_admin_evt_flt, ann_losses = bond_fct.sng_cty_bond(country=cty,
                                                                                                                                                                prot_rp=prot_rp, 
                                                                                                                                                                to_prot_share=lower_share,
                                                                                                                                                                buffer_distance_km=105,
                                                                                                                                                                res_exp=150,
                                                                                                                                                                grid_size=1000,
                                                                                                                                                                grid_specs=[5,5],
                                                                                                                                                                buffer_grid_size=5,
                                                                                                                                                                crs="EPSG:3832",
                                                                                                                                                                incl_plots=False,
                                                                                                                                                                plt_save=True,
                                                                                                                                                                storm_dir=STORM_DIR,
                                                                                                                                                                output_dir=OUTPUT_DIR,
                                                                                                                                                                ibrd_path=IBRD_DIR)
    bond_metrics_sng_dic[cty] = bond_metrics
    returns_sng_dic[cty] = returns
    premium_dic_sng_dic[cty] = premium_dic
    nominal_sng_dic[cty] = nominal
    pay_dam_df_sng_dic[cty] = pay_dam_df
    es_metrics_sng_dic[cty] = es_metrics
    int_grid_sng_dic[cty] = int_grid
    imp_per_event_flt_sng_dic[cty] = imp_per_event_flt
    imp_admin_evt_flt_sng_dic[cty] = imp_admin_evt_flt
    ann_losses_dic[cty] = ann_losses


sng_ann_ret_ibrd = {}
sng_ann_ret_regression = {}
sng_ann_ret_required = {}
sng_ann_ret_artemis = {}
sng_ann_losses = {}
nominals_sng = []
pool_tranches_ann_ret = {}
for cty in countries:
    sng_ann_ret_ibrd[cty] = returns_sng_dic[cty]['Annual'][0] 
    sng_ann_ret_regression[cty] = returns_sng_dic[cty]['Annual'][1] 
    sng_ann_ret_required[cty] = returns_sng_dic[cty]['Annual'][2] 
    sng_ann_ret_artemis[cty] = returns_sng_dic[cty]['Annual'][3] 
    sng_ann_losses[cty] = np.array(ann_losses_dic[cty]['losses'].apply(sum)) * nominal_sng_dic[cty]
    nominals_sng.append(nominal_sng_dic[cty])
sng_ann_losses = pd.DataFrame(sng_ann_losses)
sng_ann_ret_df_ibrd = pd.DataFrame(sng_ann_ret_ibrd)
sng_ann_ret_df_regression = pd.DataFrame(sng_ann_ret_regression)
sng_ann_ret_df_required = pd.DataFrame(sng_ann_ret_required)
sng_ann_ret_df_artemis = pd.DataFrame(sng_ann_ret_artemis)
es_metrics_df = pd.DataFrame(es_metrics_sng_dic)

csv_losses_name = "sng_losses.csv"
csv_metrics_name = "sng_metrics.csv"
csv_es_name = "sng_es.csv"
sng_ann_losses.to_csv(OUTPUT_DIR.joinpath(csv_losses_name), index=False, sep=',')
sng_ann_ret_df_ibrd.to_csv(OUTPUT_DIR.joinpath("sng_returns_ibrd.csv"), index=False, sep=',')
sng_ann_ret_df_regression.to_csv(OUTPUT_DIR.joinpath("sng_returns_regression.csv"), index=False, sep=',')
sng_ann_ret_df_required.to_csv(OUTPUT_DIR.joinpath("sng_returns_required.csv"), index=False, sep=',')
sng_ann_ret_df_artemis.to_csv(OUTPUT_DIR.joinpath("sng_returns_artemis.csv"), index=False, sep=',')
es_metrics_df.to_csv(OUTPUT_DIR.joinpath(csv_es_name), index=False, sep=',')

output_path = Path("/cluster/work/climate/kbergmueller/cty_data/bond_metrics_sng_dic.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as file:
    pickle.dump(bond_metrics_sng_dic, file)

output_path = Path("/cluster/work/climate/kbergmueller/cty_data/premium_dic_sng_dic.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as file:
    pickle.dump(premium_dic_sng_dic, file)

nominal_dic = {}
pay_dam_df_dic = {}
for cty in countries:
    nominal_dic[cty] = nominal_sng_dic[cty]
    pay_dam_df_dic[cty] = pay_dam_df_sng_dic[cty]

nominal_dic_df = pd.DataFrame(list(nominal_dic.items()), columns=['Key', 'Value'])
file_name = 'nominal_dic_df.csv'
nominal_dic_df.to_csv(OUTPUT_DIR.joinpath(file_name), index=False, sep=',')
# Specify the output path
output_path = Path("/cluster/work/climate/kbergmueller/cty_data/pay_dam_df_dic.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as file:
    pickle.dump(pay_dam_df_dic, file)
