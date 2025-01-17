#import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys


from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.algorithms.soo.nonconvex.ga import GA
from pooling_functions_ciullo import calc_pool_conc, pop_num, PoolOptimizationProblemFS
from pymoo.optimize import minimize
from pymoo.operators.repair.rounding import RoundingRepair



#define directories 
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")

#choose country
countries = [480, 212, 882, 332, 670, 388, 662, 214, 548, 242, 776, 174, 584]
countries_str = ['480', '212', '882', '332', '670', '388', '662', '214', '548', '242', '776', '174', '584']

sng_ann_losses = pd.read_csv(OUTPUT_DIR.joinpath("sng_losses.csv"))
sng_ann_losses = sng_ann_losses[countries_str]
tot_loss = []
tot_loss_dic = {key: [] for key in sng_ann_losses.columns}
for key in sng_ann_losses.columns:
    for i in range(len(sng_ann_losses[key])):
        tot_loss.append(sng_ann_losses[key][i])
        if len(tot_loss) == 3:
            tot_loss_dic[key].append(np.sum(tot_loss))
            tot_loss = []
tot_loss_df = pd.DataFrame(tot_loss_dic)

nominals_sng_dic = pd.read_csv(OUTPUT_DIR.joinpath("nominal_dic_df.csv"))
nominals_sng = nominals_sng_dic.set_index('Key').loc[countries, 'Value'].tolist()
max_nominal = 26000000000

#set alpha for risk diversification optimization
RT = len(tot_loss_df[key])
alpha = 1-1/RT 

n_opt_rep = 100
opt_rep = range(0,n_opt_rep,1)

def process_n(n, cntry_names, df_losses, alpha, nominals_sng, max_nominal, output_file):
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle('Convergence Plot for Risk Concentration Minimization')

    df_conc = pd.DataFrame()
    df_cntry_num = pd.DataFrame()
    df_cntry_allocation = pd.DataFrame(columns=cntry_names)
    df_result = pd.DataFrame(columns=cntry_names)
    df_result_dump = pd.DataFrame()
    
    bools = df_losses >= np.quantile(df_losses, alpha, axis=0)

    # Loop through repetitions for seed analysis
    for index in opt_rep:
        # Define Problem and Algorithm (same as inside the loop)
        problem = PoolOptimizationProblemFS(nominals_sng, max_nominal, df_losses, bools, alpha, n, calc_pool_conc)
        algorithm = GA(
            pop_size=2000,
            sampling=IntegerRandomSampling(),
            crossover=HalfUniformCrossover(),
            mutation=PolynomialMutation(repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        # Solve the problem
        res_reg = minimize(problem, algorithm, verbose=False, save_history=True)

        # Process results (same code as inside the loop)
        x = res_reg.X
        sorted_unique = sorted(set(x))
        rank_dict = {value: rank + 1 for rank, value in enumerate(sorted_unique)}
        x = [rank_dict[value] for value in x]

        # Update the allocation, concentration, and country count dataframes
        if df_cntry_allocation.empty or not (df_cntry_allocation == x).all(axis=1).any():
            df_cntry_allocation = pd.concat([df_cntry_allocation, pd.DataFrame([x], columns=cntry_names)], ignore_index=True)
            df_conc = pd.concat([df_conc, pd.DataFrame([res_reg.F], columns=['Min_Concentration'])], ignore_index=True)
            df_cntry_num = pd.concat([df_cntry_num, pd.DataFrame([np.sum(res_reg.X)], columns=['Country_Count'])], ignore_index=True)
        
        # Add to dump dataframe
        new_row = pd.DataFrame(columns=[cntry_names, 'min_conc', 'min_cntry_num', 'n_gen1', 'n_eval1_max'])
        new_row = pd.DataFrame([x], columns=cntry_names)
        new_row['min_conc'] = pd.DataFrame([res_reg.F], columns=['min_conc'])
        new_row['min_cntry_num'] = pd.DataFrame([np.sum(res_reg.X)], columns=['min_cntry_num'])
        n_evals = np.array([e.evaluator.n_eval for e in res_reg.history])
        opt = np.array([e.opt[0].F for e in res_reg.history])
        ax.plot(n_evals, opt, '--')
        ax.set_ylabel('Minimum Concentration', fontsize=12)
        ax.set_xlabel('Number of Function Evaluations', fontsize=12)

        # Add generation and evaluation max values
        new_row['n_gen1'] = pd.DataFrame([res_reg.algorithm.n_gen], columns=['n_gen1'])
        new_row['n_eval1_max'] = pd.DataFrame([n_evals.max()], columns=['n_eval1_max'])
        df_result_dump = pd.concat([df_result_dump, new_row], ignore_index=True)

    # Find minimum concentration and filter results
    min_conc = df_conc['Min_Concentration'].min()
    ind_min_conc = df_conc.index[df_conc['Min_Concentration'] == min_conc].tolist()
    ind_min = list(set(ind_min_conc))
    df_result = df_cntry_allocation.loc[ind_min].reset_index(drop=True)
    df_result.to_csv(output_file.joinpath(f"df_result_{n}_pools_fs_full_test.csv"), index=False, sep=',')
    fig.savefig(output_file.joinpath(f"convergence_plot_{n}_pools_fs_full_test.png"), dpi=300, bbox_inches='tight')
    print(f'Round {n} finished')
    return df_result, fig, min_conc

def process_pool(n):
    df_result, fig, min_conc = process_n(n, countries, tot_loss_df, alpha, nominals_sng, max_nominal, OUTPUT_DIR)
    print(df_result)
    print("Min conc: ", min_conc)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: pooling_max_nom.py <number_pools>")
        sys.exit(1)

    number_pools = int(sys.argv[1])
    process_pool(number_pools)


