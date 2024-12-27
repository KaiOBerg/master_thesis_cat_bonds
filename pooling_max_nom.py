#import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import n_fct_t_rl_thm_ll as bond_fct
from tqdm import tqdm

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.algorithms.soo.nonconvex.ga import GA
from pooling_functions_ciullo import calc_pool_conc, pop_num, PoolOptimizationProblem
from pymoo.optimize import minimize
from pymoo.operators.repair.rounding import RoundingRepair
import concurrent.futures



#define directories 
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
IBRD_DIR = Path("/cluster/work/climate/kbergmueller")

#choose country
countries = [480, 212, 332, 670, 28, 388, 52, 662, 659, 308, 214, 44, 882, 548, 242, 780, 192, 570, 84, 776, 90, 174, 184, 584, 585]

sng_ann_losses = pd.read_csv(OUTPUT_DIR.joinpath("sng_losses.csv"))
nominals_sng = pd.read_csv(OUTPUT_DIR.joinpath("nominals_sng.csv"))
max_nominal = 100000000000

#set alpha for risk diversification optimization
RT = 200
alpha = 1-1/RT 

n_opt_rep = 100
opt_rep = range(0,n_opt_rep,1)

N_arr = [1,2,3,4,5]

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
        problem = PoolOptimizationProblem(nominals_sng, max_nominal, df_losses, bools, alpha, n, calc_pool_conc)
        algorithm = GA(
            pop_size=250,
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
    df_result.to_csv(output_file.joinpath(f"df_result_{n}_pools.csv"), index=False, sep=',')
    fig.savefig(output_file.joinpath(f"convergence_plot_{n}_pools.png"), dpi=300, bbox_inches='tight')
    print(f'Round {n} finished')
    return df_result, fig, min_conc

def main():
    # Use ThreadPoolExecutor to parallelize the outer loop (over N_arr)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_n, n, countries, sng_ann_losses, alpha, nominals_sng, max_nominal, OUTPUT_DIR)
            for n in N_arr
        ]
        for future in concurrent.futures.as_completed(futures):
            result, fig, min_conc = future.result()
            print(result)
            print("Min conc: ", min_conc)

if __name__ == "__main__":
    main()


