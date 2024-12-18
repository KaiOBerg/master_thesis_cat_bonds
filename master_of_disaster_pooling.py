#import general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import dirichlet
import scipy.optimize as sco
from pathlib import Path
import n_fct_t_rl_thm_ll as bond_fct


#define directories 
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
IBRD_DIR = Path("/cluster/work/climate/kbergmueller")

#choose country
countries = [882, 480, 212, 332, 670, 28, 388, 52, 662, 659, 308, 214, 44, 548, 242, 780, 192, 570, 84, 776, 90, 174, 184, 584, 585]
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

#define tranches for pooling 
tranches_array = [np.array([100, 200, 400, 800, 2000]), np.array([100, 500, 2500]), np.array([10, 20, 40, 80, 160, 320, 640, 1280, 3000, 6000]), np.array([10, 50, 150, 300, 500, 750, 1500, 3000]), np.array([2000, 4000, 6000, 8000]), np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1500, 2000, 2500, 3000, 5000])]

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


sng_ann_ret_ibrd = {}
sng_ann_ret_regression = {}
sng_ann_ret_artemis = {}
sng_ann_losses = {}
nominals_sng = []
pool_tranches_ann_ret = {}
for cty in countries:
    sng_ann_ret_ibrd[cty] = returns_sng_dic[cty]['Annual'][0] 
    sng_ann_ret_regression[cty] = returns_sng_dic[cty]['Annual'][1] 
    sng_ann_ret_artemis[cty] = returns_sng_dic[cty]['Annual'][3] 
    sng_ann_losses[cty] = np.array(ann_losses_dic[cty]['losses'].apply(sum)) * nominal_sng_dic[cty]
    nominals_sng.append(nominal_sng_dic[cty])
sng_ann_losses = pd.DataFrame(sng_ann_losses)
sng_ann_ret_df_ibrd = pd.DataFrame(sng_ann_ret_ibrd)
sng_ann_ret_df_regression = pd.DataFrame(sng_ann_ret_regression)
sng_ann_ret_df_artemis = pd.DataFrame(sng_ann_ret_artemis)
bond_metrics_sng_dic_df = pd.concat(bond_metrics_sng_dic, names=["Key"])
bond_metrics_sng_dic_df = bond_metrics_sng_dic_df.reset_index(level=0).reset_index(drop=True)
bond_metrics_sng_dic_df.rename(columns={"Key": "Country"}, inplace=True)
es_metrics_df = pd.DataFrame(es_metrics_sng_dic)

csv_losses_name = "sng_losses.csv"
csv_metrics_name = "sng_metrics.csv"
csv_es_name = "sng_es.csv"
sng_ann_losses.to_csv(OUTPUT_DIR.joinpath(csv_losses_name), index=False, sep=',')
sng_ann_ret_df_ibrd.to_csv(OUTPUT_DIR.joinpath("sng_returns_ibrd.csv"), index=False, sep=',')
sng_ann_ret_df_regression.to_csv(OUTPUT_DIR.joinpath("sng_returns_regression.csv"), index=False, sep=',')
sng_ann_ret_df_artemis.to_csv(OUTPUT_DIR.joinpath("sng_returns_artemis.csv"), index=False, sep=',')
bond_metrics_sng_dic_df.to_csv(OUTPUT_DIR.joinpath(csv_metrics_name), index=False, sep=',')
es_metrics_df.to_csv(OUTPUT_DIR.joinpath(csv_es_name), index=False, sep=',')


nominal_dic = {}
pay_dam_df_dic = {}
for cty in countries:
    nominal_dic[cty] = nominal_sng_dic[cty]
    pay_dam_df_dic[cty] = pay_dam_df_sng_dic[cty]
    
iterator = 0
for tranches_temp in tranches_array:
    ncf_pool_tot, premiums_pool_tot, premium_dic_pool_tot, nominal_pool_tot, es_metrics_pool_tot, MES_cty_pool_tot, tranches_tot = bond_fct.mlt_cty_bond(countries=countries,
                                                                                                                                                         pay_dam_df_dic=pay_dam_df_dic,
                                                                                                                                                         nominals_dic=nominal_dic, 
                                                                                                                                                         tranches_array=tranches_temp, 
                                                                                                                                                         opt_cap=True,
                                                                                                                                                         ibrd_path=IBRD_DIR)

    pool_tranches_ann_ret_df_ibrd = pd.DataFrame(ncf_pool_tot['ibrd'])
    pool_tranches_ann_ret_df_regression = pd.DataFrame(ncf_pool_tot['regression'])
    pool_tranches_ann_ret_df_artemis = pd.DataFrame(ncf_pool_tot['artemis'])
    pool_tranches_ann_ret_df_ibrd = pool_tranches_ann_ret_df_ibrd.drop('Total', axis=1)
    pool_tranches_ann_ret_df_regression = pool_tranches_ann_ret_df_regression.drop('Total', axis=1)
    pool_tranches_ann_ret_df_artemis = pool_tranches_ann_ret_df_artemis.drop('Total', axis=1)
    for col in pool_tranches_ann_ret_df_ibrd.columns:
        pool_tranches_ann_ret_df_ibrd[col] = pool_tranches_ann_ret_df_ibrd[col] / tranches_tot.loc[tranches_tot['RP'] == col, 'nominal'].iloc[0]
        pool_tranches_ann_ret_df_regression[col] = pool_tranches_ann_ret_df_regression[col] / tranches_tot.loc[tranches_tot['RP'] == col, 'nominal'].iloc[0]
        pool_tranches_ann_ret_df_artemis[col] = pool_tranches_ann_ret_df_artemis[col] / tranches_tot.loc[tranches_tot['RP'] == col, 'nominal'].iloc[0]

    pool_ann_ret_ibrd = ncf_pool_tot['ibrd']['Total']
    pool_ann_ret_regression = ncf_pool_tot['regression']['Total']
    pool_ann_ret_artemis = ncf_pool_tot['artemis']['Total']

    pool_tranches_ann_ret_df_ibrd.to_csv(OUTPUT_DIR.joinpath(f"tranches_{str(iterator)}_returns_ibrd.csv"), index=False, sep=',')
    pool_tranches_ann_ret_df_regression.to_csv(OUTPUT_DIR.joinpath("tranches_{str(iterator)}_returns_regression.csv"), index=False, sep=',')
    pool_tranches_ann_ret_df_artemis.to_csv(OUTPUT_DIR.joinpath("tranches_{str(iterator)}_returns_artemis.csv"), index=False, sep=',')


    for prem_mode in ['ibrd', 'regression', 'artemis']: 
        if prem_mode == 'ibrd':
            sng_ann_ret = sng_ann_ret_ibrd
            sng_ann_ret_df = sng_ann_ret_df_ibrd
            pool_tranches_ann_ret_df = pool_tranches_ann_ret_df_ibrd
            pool_ann_ret = pool_ann_ret_ibrd
            sng_cty_premium = []
            sng_cty_pay = []
            for cty in bond_metrics_sng_dic:
                sng_cty_premium.append(bond_metrics_sng_dic[cty]['Total Premiums'][0])
                sng_cty_pay.append(bond_metrics_sng_dic[cty]['Summed Payments'][0])

        elif prem_mode == 'regression':
            sng_ann_ret = sng_ann_ret_regression
            sng_ann_ret_df = sng_ann_ret_df_regression
            pool_tranches_ann_ret_df = pool_tranches_ann_ret_df_regression
            pool_ann_ret = pool_ann_ret_regression
            sng_cty_premium = []
            sng_cty_pay = []
            for cty in bond_metrics_sng_dic:
                sng_cty_premium.append(bond_metrics_sng_dic[cty]['Total Premiums'][1])
                sng_cty_pay.append(bond_metrics_sng_dic[cty]['Summed Payments'][1])

        elif prem_mode == 'artemis':
            sng_ann_ret = sng_ann_ret_artemis
            sng_ann_ret_df = sng_ann_ret_df_artemis
            pool_tranches_ann_ret_df = pool_tranches_ann_ret_df_artemis
            pool_ann_ret = pool_ann_ret_artemis
            sng_cty_premium = []
            sng_cty_pay = []
            for cty in bond_metrics_sng_dic:
                sng_cty_premium.append(bond_metrics_sng_dic[cty]['Total Premiums'][3])
                sng_cty_pay.append(bond_metrics_sng_dic[cty]['Summed Payments'][3])

        else:
            print('Wrong input premium mode')
            break

        df_returns = pd.DataFrame({f"{country} Returns": returns for country, returns in sng_ann_ret.items()})
        df_returns_pool = pd.DataFrame({f"{country} Returns": returns for country, returns in pool_tranches_ann_ret_df.items()})

        r = np.mean(sng_ann_ret_df,axis=0)
        r_pool= np.mean(pool_tranches_ann_ret_df,axis=0)

        # Create a covariance matrix
        covar = sng_ann_ret_df.cov()
        covar_pool = pool_tranches_ann_ret_df.cov()

        p_ret = [] # Define an empty array for portfolio returns
        p_vol = [] # Define an empty array for portfolio volatility
        p_weights = [] # Define an empty array for asset weights

        num_assets = len(sng_ann_ret_df.columns)

        p_ret_pool = [] # Define an empty array for portfolio returns
        p_vol_pool = [] # Define an empty array for portfolio volatility
        p_weights_pool = [] # Define an empty array for asset weights

        num_assets_pool = len(pool_tranches_ann_ret_df.columns)

        num_portfolios = 10000  # Number of portfolios to simulate
        alpha_port = 0.5


        for _ in range(num_portfolios):
            weights = dirichlet([alpha_port] * num_assets)
            weights = weights/np.sum(weights)
            p_weights.append(weights)
            returns = np.dot(weights, r) 
            p_ret.append(returns)
            var = covar.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
            sd = np.sqrt(var) # yearly standard deviation
            p_vol.append(sd)    

            weights_pool = dirichlet([alpha_port] * num_assets_pool)
            weights_pool = weights_pool/np.sum(weights_pool)
            p_weights_pool.append(weights_pool)
            returns_pool = np.dot(weights_pool, r_pool) 
            p_ret_pool.append(returns_pool)
            var_pool = covar_pool.mul(weights_pool, axis=0).mul(weights_pool, axis=1).sum().sum()# Portfolio Variance
            sd_pool = np.sqrt(var_pool) # yearly standard deviation
            p_vol_pool.append(sd_pool)


        data = {'Returns':p_ret, 'Volatility':p_vol, 'Sharpe Ratio':np.array(p_ret)/np.array(p_vol)}
        data_pool = {'Returns':p_ret_pool, 'Volatility':p_vol_pool, 'Sharpe Ratio':np.array(p_ret_pool)/np.array(p_vol_pool)}

        for counter, symbol in enumerate(sng_ann_ret_df.columns.tolist()):
            data[str(symbol)+' weight'] = [w[counter] for w in p_weights]

        for counter, symbol in enumerate(pool_tranches_ann_ret_df.columns.tolist()):
            data_pool[str(symbol)+' weight'] = [w[counter] for w in p_weights_pool]


        portfolios  = pd.DataFrame(data)
        max_sharpe_idx = portfolios['Sharpe Ratio'].idxmax()
        max_sharpe_portfolio = portfolios.loc[max_sharpe_idx]
        portfolios.head() 

        portfolios_pool  = pd.DataFrame(data_pool)
        max_sharpe_idx_pool = portfolios_pool['Sharpe Ratio'].idxmax()
        max_sharpe_portfolio_pool = portfolios_pool.loc[max_sharpe_idx_pool]


        # Expected returns and covariance matrices
        r = np.mean(df_returns, axis=0)  # Expected returns for df_returns
        r_pool = np.mean(df_returns_pool, axis=0)  # Expected returns for df_returns_pool

        covar = df_returns.cov()  # Covariance matrix for df_returns
        covar_pool = df_returns_pool.cov()  # Covariance matrix for df_returns_pool

        # Function to calculate portfolio risk (standard deviation)
        def portfolio_risk(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Function to calculate portfolio return
        def portfolio_return(weights, mean_returns):
            return np.sum(weights * mean_returns)

        # Function for optimization: minimize risk for a given return
        def minimize_risk(target_return, mean_returns, cov_matrix):
            # Number of assets in the portfolio
            num_assets = len(mean_returns)

            # Constraints: weights sum to 1, and portfolio return equals target return
            constraints = ({
                'type': 'eq', 'fun': lambda w: np.sum(w) - 1  # Weights sum to 1
            }, {
                'type': 'eq', 'fun': lambda w: portfolio_return(w, mean_returns) - target_return  # Target return constraint
            })

            # Initial guess: equal weights
            initial_guess = np.ones(num_assets) / num_assets

            # Bounds for weights: between 0 and 1
            bounds = tuple((0, 1) for asset in range(num_assets))

            # Minimize risk (objective function)
            result = sco.minimize(portfolio_risk, initial_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)

            return result.x, result.fun  # Return the optimized weights and the minimized risk

        # Generate a series of target returns and corresponding risks
        target_returns = np.linspace(np.min(r), np.max(r), 50)  # Range of target returns
        risks = []

        for target_return in target_returns:
            weights, risk = minimize_risk(target_return, r, covar)
            risks.append(risk)

        ## Repeat for the second data set (df_returns_pool)
        risks_pool = []
        target_returns_pool = np.linspace(np.min(r_pool), np.max(r_pool), 50)
        for target_return in target_returns_pool:
            weights, risk = minimize_risk(target_return, r_pool, covar_pool)
            risks_pool.append(risk)


        # Plot efficient frontier
        plt_name = f"risk_return_frontier_{prem_mode}_{str(iterator)}.png"
        plt.figure(figsize=[10,10])

        plt.scatter(x=portfolios['Volatility'], y=portfolios['Returns'], c=portfolios['Sharpe Ratio'], cmap='cividis', marker='v', s=10, alpha=0.3)
        plt.text(max_sharpe_portfolio['Volatility'] + 0.001,max_sharpe_portfolio['Returns'],f"Max Sharpe: {max_sharpe_portfolio['Sharpe Ratio']:.2f}",fontsize=10,ha='left',va='center',color='red')
        plt.plot(risks, target_returns, label="Efficient Frontier - Sng", color='blue')
        plt.scatter(x=portfolios_pool['Volatility'], y=portfolios_pool['Returns'], c=portfolios_pool['Sharpe Ratio'], cmap='cividis', marker='o', s=10, alpha=0.3)
        plt.text(max_sharpe_portfolio_pool['Volatility'] + 0.001,max_sharpe_portfolio_pool['Returns'],f"Max Sharpe: {max_sharpe_portfolio_pool['Sharpe Ratio']:.2f}",fontsize=10,ha='left',va='center',color='red')
        plt.scatter(x=max_sharpe_portfolio_pool['Volatility'], y=max_sharpe_portfolio_pool['Returns'], color='red', marker='x', s=10, alpha=1.0)
        plt.scatter(x=max_sharpe_portfolio['Volatility'], y=max_sharpe_portfolio['Returns'], color='red', marker='x', s=10, alpha=1.0)
        plt.plot(risks_pool, target_returns_pool, label="Efficient Frontier - Pool", color='red')

        # Plot pool point
        plt.scatter(np.std(pool_ann_ret), np.mean(pool_ann_ret), label='Pool', color='purple', s=100)
        plt.text(np.std(pool_ann_ret)-0.01,np.mean(pool_ann_ret),f'Sharpe: {np.mean(pool_ann_ret)/np.std(pool_ann_ret):.2f}',fontsize=10,ha='right',va='center',color='purple')

        for cty in countries:
            plt.scatter(np.std(sng_ann_ret[cty]), np.mean(sng_ann_ret[cty]), label=cty, s=100)
        plt.xlabel("Volatility")
        plt.ylabel("Returns")
        plt.grid(True)
        plt.legend()
        plt.savefig(OUTPUT_DIR.joinpath(plt_name))


        x = countries.copy()
        x = [str(entry) for entry in x]
        x.append('Total')
        y = (np.array(sng_cty_premium)/np.array(sng_cty_pay)).tolist()
        y.append(np.sum(premiums_pool_tot['regression']['Total_alt'])/(es_metrics_pool_tot['Payout']/nominal_pool_tot))

        plt_name = f"insurance_multiple_{prem_mode}_{str(iterator)}.png"
        plt.figure(figsize=[10,10])
        plt.scatter(x,y)
        plt.xlabel("Country")
        plt.ylabel("Insurance multiple")
        plt.grid(True)
        plt.savefig(OUTPUT_DIR.joinpath(plt_name))


        s_pool = []
        n = []
        s = sng_cty_premium
        for cty in countries:
            s_pool.append(np.sum(premiums_pool_tot['regression'][cty])*nominal_pool_tot)
            n.append(nominal_sng_dic[cty])

        prem_diff = (np.array(s_pool)/np.array(s)).tolist()
        prem_diff.append(float(np.sum(premiums_pool_tot['regression']['Total_alt'])*nominal_pool_tot/np.sum(s)))
        prem_diff.append(float(np.sum(premiums_pool_tot['regression']['Total'])*nominal_pool_tot/np.sum(s)))

        country_str = [str(entry) for entry in countries]
        country_str.append('Total_alt')
        country_str.append('Total')
        print(prem_diff)

        plt_name = f"premium_savings_{prem_mode}_{str(iterator)}.png"
        plt.figure(figsize=[10,10])
        plt.scatter(country_str, prem_diff)
        plt.xlabel("Bonds")
        plt.ylabel("Premium savings")
        plt.grid(True)
        plt.savefig(OUTPUT_DIR.joinpath(plt_name))
        iterator += 1