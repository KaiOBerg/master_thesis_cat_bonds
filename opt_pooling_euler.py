#import general packages
import numpy as np
from pathlib import Path

import n_fct_t_rl_thm_ll as bond_fct
import copy
import pandas as pd

countries_30 = [585, 583, 584, 882, 776, 798, 570, 184, 480, 690, 174, 462, 780, 670, 662, 659, 52, 308, 28, 212, 132]
countries_150 = [548, 626, 388, 332, 214, 192] #90, 44


#define minimum return period to be covered
lower_rp = 0.05
lower_share = 0.045
#define maximum return period to be covered
upper_share = 0.1

#define benchmark sharpe ratio
target_sharpe = 0.5

#define the risk free rate
rf_rate = 0.00

OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller")
HAZARD_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")

bond_cache = {}
bond_metrics_dic = {}
returns_dic = {}
premium_dic_dic = {}
nominal_dic = {}
pay_dam_df_dic = {}
es_metrics_dic = {}
int_grid_dic = {}
imp_per_event_flt_dic = {}
imp_admin_evt_flt_dic = {}

def get_bond_metrics(pool, pay_dam_pool_it, nominal_pool_it):
    pool_key = tuple(sorted(pool))  # Create a unique key for the pool
    if pool_key not in bond_cache:
        # If result isn't cached, compute and store it
        pay_dam_temp = {c: pay_dam_pool_it[c] for c in pool}
        nominal_temp = {c: nominal_pool_it[c] for c in pool}
        bond_metrics, returns, tot_coverage, premium_dic, nominal, es_metrics, MES_cty = bond_fct.mlt_cty_bond(
            countries=pool,
            pay_dam_df_dic=pay_dam_temp,
            nominals_dic=nominal_temp,
            rf_rate=rf_rate,
            target_sharpe=target_sharpe,
            opt_cap=True,
            ibrd_path=OUTPUT_DIR,
        )
        bond_cache[pool_key] = {
            "ibrd": premium_dic["ibrd"] * nominal,
            "regression": premium_dic["regression"] * nominal,
            }
    return bond_cache[pool_key]


def create_sng_bonds(countries_30, countries_150):
    countries = []
    for cty in countries_30:
        try:
            print('Creating cat bond for ',cty)
            bond_metrics, returns, premium_dic, nominal, pay_dam_df, es_metrics, int_grid, imp_per_event_flt, imp_admin_evt_flt = bond_fct.sng_cty_bond(country=cty,
                                                                                                                                                        prot_share=upper_share, 
                                                                                                                                                        rf_rate=rf_rate, 
                                                                                                                                                        target_sharpe=target_sharpe, 
                                                                                                                                                        to_prot_share=lower_share,
                                                                                                                                                        buffer_distance_km=105,
                                                                                                                                                        res_exp=30,
                                                                                                                                                        grid_size=6000,
                                                                                                                                                        output_dir=HAZARD_DIR,
                                                                                                                                                        storm_dir=STORM_DIR,
                                                                                                                                                        ibrd_path=OUTPUT_DIR)
            bond_metrics_dic[cty] = bond_metrics
            returns_dic[cty] = returns
            premium_dic_dic[cty] = premium_dic
            nominal_dic[cty] = nominal
            pay_dam_df_dic[cty] = pay_dam_df
            es_metrics_dic[cty] = es_metrics
            int_grid_dic[cty] = int_grid
            imp_per_event_flt_dic[cty] = imp_per_event_flt
            imp_admin_evt_flt_dic[cty] = imp_admin_evt_flt

            print(f'Single Country Bond finished: {cty}')
            countries.append(cty)

        except Exception as e: 
            print(f"Error processing country {cty}: {e}")

    for cty in countries_150:
        try:
            print('Creating cat bond for ',cty)
            bond_metrics, returns, premium_dic, nominal, pay_dam_df, es_metrics, int_grid, imp_per_event_flt, imp_admin_evt_flt = bond_fct.sng_cty_bond(country=cty,
                                                                                                                                                        prot_share=upper_share, 
                                                                                                                                                        rf_rate=rf_rate, 
                                                                                                                                                        target_sharpe=target_sharpe, 
                                                                                                                                                        to_prot_share=lower_share,
                                                                                                                                                        buffer_distance_km=105,
                                                                                                                                                        res_exp=150,
                                                                                                                                                        grid_size=6000,
                                                                                                                                                        output_dir=HAZARD_DIR,
                                                                                                                                                        storm_dir=STORM_DIR,
                                                                                                                                                        ibrd_path=OUTPUT_DIR)




            bond_metrics_dic[cty] = bond_metrics
            returns_dic[cty] = returns
            premium_dic_dic[cty] = premium_dic
            nominal_dic[cty] = nominal
            pay_dam_df_dic[cty] = pay_dam_df
            es_metrics_dic[cty] = es_metrics
            int_grid_dic[cty] = int_grid
            imp_per_event_flt_dic[cty] = imp_per_event_flt
            imp_admin_evt_flt_dic[cty] = imp_admin_evt_flt

            print(f'Single Country Bond finished: {cty}')
            countries.append(cty)
        except Exception as e:
            print(f"Error processing country {cty}: {e}")

    return bond_metrics_dic, returns_dic, premium_dic_dic, nominal_dic, pay_dam_df_dic, es_metrics_dic, int_grid_dic, imp_per_event_flt_dic, imp_admin_evt_flt_dic, countries



if __name__ == "__main__":
    bond_metrics_dic, returns_dic, premium_dic_dic, nominal_dic, pay_dam_df_dic, es_metrics_dic, int_grid_dic, imp_per_event_flt_dic, imp_admin_evt_flt_dic, countries = create_sng_bonds(countries_30, countries_150)

    pool_comb = pd.DataFrame(columns = ['Pool', 'Premium_ibrd', 'Premium_regr'])
    pay_dam_pool_it = {}
    nominal_pool_it = {}
    for i in range(len(countries)):
        pay_dam_pool_it[countries[i]] = pay_dam_df_dic[countries[i]]
        nominal_pool_it[countries[i]] = nominal_dic[countries[i]]

    bond_metrics_pool, returns_pool, tot_coverage_cty_pool, premium_dic_pool, nominal_pool, es_metrics_pool, MES_cty_pool = bond_fct.mlt_cty_bond(countries=countries, pay_dam_df_dic=pay_dam_pool_it, nominals_dic=nominal_pool_it, rf_rate=rf_rate, target_sharpe=target_sharpe, opt_cap=False, ibrd_path=OUTPUT_DIR)
    tmp_abs_prem = abs_prem = premium_dic_pool['ibrd'] * nominal_pool
    tmp_abs_prem_regr = premium_dic_pool['regression'] * nominal_pool
    pool_comb.loc[len(pool_comb)] = [countries, tmp_abs_prem, tmp_abs_prem_regr]


    countries_main_pool = countries.copy()


    iteration = 0
    country_side_pools = []
    while tmp_abs_prem <= abs_prem and iteration < len(countries)-1:
        iteration += 1 
        tmp_abs_prem = abs_prem
        prem_placeholder = np.inf
        for i in range(len(countries_main_pool)):
            cty = countries_main_pool[i]
            tmp_main_pool = countries_main_pool.copy()
            tmp_main_pool.pop(i)
            main_pool_metrics = get_bond_metrics(tmp_main_pool, pay_dam_pool_it, nominal_pool_it)
            tmp_abs_prem_main = main_pool_metrics["ibrd"]
            tmp_abs_prem_main_regr = main_pool_metrics["regression"]
            if country_side_pools:
                for inner_pool in range(len(country_side_pools)):
                    tmp_abs_prem_ls = []
                    tmp_side_pools = copy.deepcopy(country_side_pools)
                    tmp_side_pools[inner_pool].append(cty)
                    tmp_abs_prem_ls = [get_bond_metrics(pool, pay_dam_pool_it, nominal_pool_it)["ibrd"] for pool in tmp_side_pools]
                    tmp_abs_prem_ls_regr = [get_bond_metrics(pool, pay_dam_pool_it, nominal_pool_it)["regression"] for pool in tmp_side_pools]
                    tot_prem_it = np.sum(tmp_abs_prem_ls) + tmp_abs_prem_main
                    tot_prem_it_regr = np.sum(tmp_abs_prem_ls_regr) + tmp_abs_prem_main_regr
                    if tot_prem_it < prem_placeholder:
                        prem_placeholder = tot_prem_it
                        tmp_country_side_pools = tmp_side_pools
                        tmp_country_main_pool = tmp_main_pool
                        tmp_abs_prem_regr = tot_prem_it_regr
            tmp_side_pools = copy.deepcopy(country_side_pools)
            tmp_side_pools.append([cty])
            tmp_abs_prem_ls = [get_bond_metrics(pool, pay_dam_pool_it, nominal_pool_it)["ibrd"] for pool in tmp_side_pools]
            tmp_abs_prem_ls_regr = [get_bond_metrics(pool, pay_dam_pool_it, nominal_pool_it)["regression"] for pool in tmp_side_pools]
            tot_prem_it = np.sum(tmp_abs_prem_ls) + tmp_abs_prem_main
            tot_prem_it_regr = np.sum(tmp_abs_prem_ls_regr) + tmp_abs_prem_main_regr
            if tot_prem_it < prem_placeholder:
                prem_placeholder = tot_prem_it
                tmp_country_side_pools = tmp_side_pools
                tmp_country_main_pool = tmp_main_pool
                tmp_abs_prem_regr = tot_prem_it_regr
        tmp_abs_prem = prem_placeholder
        country_side_pools = tmp_country_side_pools
        countries_main_pool = tmp_country_main_pool
        pool_comb.loc[len(pool_comb)] = [[countries_main_pool,country_side_pools], tmp_abs_prem, tmp_abs_prem_regr]

    output_file = OUTPUT_DIR / "opt_pool.xlsx"
    pool_comb.to_excel(output_file, index=False)
    print(f"Saved results to {output_file}")
