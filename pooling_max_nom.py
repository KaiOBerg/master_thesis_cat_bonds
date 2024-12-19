#import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import n_fct_t_rl_thm_ll as bond_fct

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.algorithms.soo.nonconvex.ga import GA
from pooling_functions_ciullo import calc_pool_conc, PoolOptimizationProblem
from pymoo.optimize import minimize
from pymoo.operators.repair.rounding import RoundingRepair


#define directories 
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
IBRD_DIR = Path("/cluster/work/climate/kbergmueller")

#choose country
countries = [480, 212, 332, 670, 28, 388, 52, 662, 659, 308, 214, 44, 882, 548, 242, 780, 192, 570, 84, 776, 90, 174, 184, 584, 585]
countries_150 = [332, 388, 214, 44, 548, 192, 84, 90] 
fiji = [242]
countries_30 = [480, 212, 670, 28, 52, 662, 659, 308, 882, 780, 570, 776, 174, 184, 584, 585]

#set risk free rate, either single value or array
rf_rates = 0.00
#set risk muliplier reported by artems
artemis_multiplier = 4.11
#set sharpe ratio to beat
target_sharpe = 0.5
#define bond setting
lower_share = 0.045
prot_rp = 250

#set alpha for risk diversification optimization
RT = 500
alpha = 1-1/RT 

#set max nominal for pool
max_nominal = 6720000000

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
                                                                                                                                                                grid_size=10000,
                                                                                                                                                                grid_specs=[3,3],
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
                                                                                                                                                                grid_size=10000,
                                                                                                                                                                grid_specs=[1,1],
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
                                                                                                                                                                grid_size=10000,
                                                                                                                                                                grid_specs=[1,1],
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

sng_ann_losses = {}
nominals_sng = []
for cty in countries:
    sng_ann_losses[cty] = np.array(ann_losses_dic[cty]['losses'].apply(sum)) * nominal_sng_dic[cty]
    nominals_sng.append(nominal_sng_dic[cty])
sng_ann_losses = pd.DataFrame(sng_ann_losses)
df_nominals_sng = pd.DataFrame(nominals_sng, columns=['Nominals'])
sng_ann_losses.to_csv(OUTPUT_DIR.joinpath("sng_ann_losses_pooling.csv"), index=False, sep=',')
df_nominals_sng.to_csv(OUTPUT_DIR.joinpath("nominals_sng.csv"), index=False, sep=',')


from tqdm import tqdm
#set number of repetitions of both minimizations (seed analysis)
n_opt_rep = 50
opt_rep = range(0,n_opt_rep,1)

fig, ax = plt.subplots(1, 1, figsize=(10,5))
fig.suptitle('Convergence Plot for Risk Concentration Minimization (Step 1)')
#load names of countries in region which is optimized in
df_losses = sng_ann_losses.copy()
cntry_names = countries      
#optimize for each region to derive regional pool
#empty DataFrames to fill along the way (per region and GCM -> for each GCM seperate code file or change k)
#minimized concentration and number of countries (result of 1.Opt-Step and 2. Opt-Step)
df_conc = pd.DataFrame()
df_cntry_num = pd.DataFrame()
df_cntry_allocation = pd.DataFrame(columns = cntry_names)
#end result only with solutions with min_conc and min_cntry_num
df_result = pd.DataFrame(columns = cntry_names)
#dump with all results derived for all repetitions (--> cntry_allocation, min_conc, min_cntr, (n_gen, max_n_evals for each rep) saved)
df_result_dump = pd.DataFrame()
#Bool df where > VAR200
bools = df_losses >= np.quantile(df_losses, alpha, axis=0)
#loop through repetitions for seed analysis
for index in tqdm(opt_rep, desc=f'Repetitions'):
    #Problem definition
    problem = PoolOptimizationProblem(nominals_sng, max_nominal, df_losses, bools, alpha, calc_pool_conc)

    algorithm = GA(
        pop_size=2**20,
        sampling = IntegerRandomSampling(),
        crossover = HalfUniformCrossover(),
        mutation = PolynomialMutation(repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    # Solve the problem
    res_reg = minimize(problem,
                       algorithm,
                       verbose=False,
                       save_history=True)
    
    #Indices for full regional country list
    x = res_reg.X
    print(x)
    sorted_unique = sorted(set(x))
    rank_dict = {value: rank + 1 for rank, value in enumerate(sorted_unique)}
    x = [rank_dict[value] for value in x]

    #test if country composition was already derived in previous run, if yes not appended, in not appended as new solution
    if df_cntry_allocation.empty or not (df_cntry_allocation == x).all(axis=1).any():
        df_cntry_allocation = pd.concat([df_cntry_allocation, pd.DataFrame([x], columns=cntry_names)], ignore_index=True)
        df_conc = pd.concat([df_conc, pd.DataFrame([res_reg.F], columns=['Min_Concentration'])], ignore_index=True)
        df_cntry_num = pd.concat([df_cntry_num, pd.DataFrame([np.sum(res_reg.X)], columns=['Country_Count'])], ignore_index=True)
    else:
        None

    #new_row to dump all results with all specs derived in the two optimization steps
    new_row = pd.DataFrame(columns=[cntry_names, 
                                    'min_conc', 
                                    'min_cntry_num', 
                                    'n_gen1', 
                                    'n_eval1_max'])
    new_row = pd.DataFrame([x], columns=cntry_names)
    new_row['min_conc'] = pd.DataFrame([res_reg.F], columns = ['min_conc'])
    new_row['min_cntry_num'] = pd.DataFrame([np.sum(res_reg.X)], columns=['min_cntry_num'])
    #input for convergence plot per repetition for res_reg
    n_evals = np.array([e.evaluator.n_eval for e in res_reg.history])
    opt = np.array([e.opt[0].F for e in res_reg.history])
    ax.plot(n_evals, opt, '--')
    ax.set_ylabel('Minimum Concentration')
    ax.set_xlabel('Number of Function Evaluations')

    #convergence results for new_row (dump) from res_reg
    new_row['n_gen1'] = pd.DataFrame([res_reg.algorithm.n_gen], columns = ['n_gen1'])
    new_row['n_eval1_max'] = pd.DataFrame([n_evals.max()], columns=['n_eval1_max'])
    #append new_row to dump
    df_result_dump = pd.concat([df_result_dump, new_row], ignore_index=True)

min_conc = df_conc['Min_Concentration'].min()
#find indices where conc minimized (constraint for opt step 1)
ind_min_conc = df_conc.index[df_conc['Min_Concentration'] == min_conc].tolist() 

ind_min = list(set(ind_min_conc))

#reduce full df with cntry allocations to the rows where constraint 1 is reached
df_result = df_cntry_allocation.loc[ind_min].reset_index(drop=True)
df_result.to_csv(OUTPUT_DIR.joinpath("pooling_results.csv"), index=False, sep=',')
fig.savefig(OUTPUT_DIR.joinpath("convergence_plot.png"), dpi=300, bbox_inches='tight')
print('round finished')

