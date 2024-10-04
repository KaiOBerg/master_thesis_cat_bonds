import numpy as np
import matplotlib.pyplot as plt
import math

def init_pre_plot(countries, premium_dic, nom_arr_dic, premium_dic_pool, nominal_arr_pool, MES_cty_dic, es_metrics_pool_dic, prot_share_arr, prc_mth='ibrd', target_sharpe='0.5'):    
    sng_bonds = {
        country: {
            'ibrd': [],
            'regression': [],
            'artemis': [],
            'required': [],
            'exp_loss': [],
            'abs_prem': []
        }
        for country in countries
    }

    mlty_bond_cty = {
        country: {
            'abs_prem_95': [],
            'abs_prem_99': [],
            'diff_prem_95': []
        }
        for country in countries
    }

    mlt_bond = {
        'ibrd': [],
        'regression': [],
        'artemis': [],
        'required': [],
        'exp_loss': []
    }
    
    for cty in countries: 
        for ps_share in prot_share_arr:
            ps_str = str(round(ps_share),2)
    
            sng_bonds[cty]['artemis'].append(premium_dic[cty][ps_str]['artemis'])
            sng_bonds[cty]['ibrd'].append(premium_dic[cty][ps_str]['ibrd'])
            sng_bonds[cty]['regression'].append(premium_dic[cty][ps_str]['regression'])
            sng_bonds[cty]['required'].append(premium_dic[cty][ps_str]['required'])
            sng_bonds[cty]['exp_loss'].append(premium_dic[cty][ps_str]['exp_loss'])

            mlty_bond_cty[cty]['abs_prem_95'].append(MES_cty_dic[ps_str][cty]['95'] / es_metrics_pool_dic[ps_str]['ES_95_ann'] * premium_dic_pool[ps_str][prc_mth] * nominal_arr_pool[i])
            mlty_bond_cty[cty]['abs_prem_99'].append(MES_cty_dic[ps_str][cty]['99'] / es_metrics_pool_dic[ps_str]['ES_99_ann'] * premium_dic_pool[ps_str][prc_mth] * nominal_arr_pool[i])

        sng_bonds[cty]['abs_prem']  = [e1 * e2 for e1, e2 in zip(sng_bonds[cty][prc_mth], nom_arr_dic[cty])]
        mlty_bond_cty[cty]['diff_prem_95']  = [(e1 - e2) / e1 for e1, e2 in zip(sng_bonds[cty]['abs_prem'] , mlty_bond_cty[cty]['abs_prem_95'])]


    abs_prem_sng = []
    nominal_pool = []
    for i in range(len(sng_bonds[cty]['abs_prem'])):
        premiums = []
        noms = []
        for cty in sng_bonds:
            premiums.append(sng_bonds[cty]['abs_prem'][i])
            noms.append(nom_arr_dic[cty][i])
        abs_prem_sng.append(np.sum(premiums))  
        nominal_pool.append(np.sum(noms))  
    
    mlt_bond['required'].append(premium_dic_pool[ps_str]['required'])
    mlt_bond['ibrd'].append(premium_dic_pool[ps_str]['ibrd'])
    mlt_bond['artemis'].append(premium_dic_pool[ps_str]['artemis'])
    mlt_bond['regression'].append(premium_dic_pool[ps_str]['regression'])
    mlt_bond['exp_loss'].append(premium_dic_pool[ps_str]['exp_loss'])
    
    abs_prem_pool  = [e1 * e2 for e1, e2 in zip(mlt_bond[prc_mth], nominal_arr_pool)]
    diff_prem  = [(e1 - e2) / e1 for e1, e2 in zip(abs_prem_sng, abs_prem_pool)]
    
    
    num_countries = len(sng_bonds.keys())
    countries = list(sng_bonds.keys())  # Convert keys to a list

    # Define the number of rows and columns for the subplots
    n_cols = num_countries  # Choose the number of columns (for a wide layout)
    n_rows = math.ceil(num_countries+4 / n_cols)  # Determine the number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 6 * n_rows))

    # Flatten the axes array in case of more than 1 row
    if n_rows > 1:
        axes = axes.flatten()

    # Iterate over the countries and plot each one in its respective subplot
    for i, cty in enumerate(countries):
        ax = axes[i]
        ax.plot(prot_share_arr, sng_bonds[cty]['artemis'], label='artemis')
        ax.plot(prot_share_arr, sng_bonds[cty]['ibrd'], label='ibrd')
        ax.plot(prot_share_arr, sng_bonds[cty]['regression'], label='regression')
        ax.plot(prot_share_arr, sng_bonds[cty]['required'], label=f'SR={target_sharpe}')
        ax.set_xlabel('Nominal [share of GDP]')
        ax.set_ylabel('Premium [share of nominal]')
        ax.set_title(f'Relative Premium - Country {countries[i]}')
        ax.legend()
    
    ax = axes[i+1]
    ax.plot(prot_share_arr, mlt_bond['artemis'], label='artemis')
    ax.plot(prot_share_arr, mlt_bond['ibrd'], label='ibrd')
    ax.plot(prot_share_arr, mlt_bond['regression'], label='regression')
    ax.plot(prot_share_arr, mlt_bond['required'], label=f'SR={target_sharpe}')
    ax.set_xlabel('Nominal [share of GDP]')
    ax.set_ylabel('Premium [share of nominal]')
    ax.set_title('Relative Premium - Pooled')
    ax.legend()

    ax = axes[i+2]
    ax.plot(prot_share_arr, abs_prem_sng, label=f'Single: {prc_mth} - Pricing')
    ax.plot(prot_share_arr, abs_prem_pool, label=f'Pooled: {prc_mth} - Pricing')
    ax.set_xlabel('Nominal [share of GDP]')
    ax.set_ylabel('Premium [USD]')
    ax.set_title('What premiums are needed to protect % GDP for each country?')
    ax.legend()

    color='tab:red'
    ax_1 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    ax_1.set_ylabel('Saved Premiums []', color=color)  # we already handled the x-label with ax1
    ax_1.plot(prot_share_arr, diff_prem, linestyle='dotted', label='Saved premiums', color=color)
    ax_1.tick_params(axis='y', labelcolor=color)

    lines1, labels1 = ax.get_legend_handles_labels()  
    lines2, labels2 = ax_1.get_legend_handles_labels()  

    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax = axes[i+3]
    ax.plot(nominal_pool, abs_prem_sng, label=f'Single: {prc_mth} - Pricing')
    ax.plot(nominal_arr_pool, abs_prem_pool, label=f'Pooled: {prc_mth} - Pricing')
    ax.set_xlabel('Nominal [USD]')
    ax.set_ylabel('Premium [USD]')
    ax.set_xlim(0,np.max(nominal_arr_pool))
    ax.set_title('What nominal do I get for the same premium?')
    ax.legend()

    ax = axes[i+4]
    for cty in countries:
        ax.plot(prot_share_arr, mlty_bond_cty[cty]['abs_prem_95'], label=f'Multi - Country {cty}')
        ax.plot(prot_share_arr, mlty_bond_cty[cty]['diff_prem_95'], label=f'Single: {prc_mth} - Pricing')
        ax.plot(prot_share_arr, sng_bonds[cty]['abs_prem'], label=f'Single - Country {cty}')
    ax.set_xlabel('Nominal [share of GDP]')
    ax.set_ylabel('Premium [USD]')
    ax.set_title('Premiums - Bonds - Country Specific')

    color='tab:red'
    ax_2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    ax_2.set_ylabel('Saved Premiums []', color=color)  # we already handled the x-label with ax1
    for cty in countries:
        ax.plot(prot_share_arr, mlty_bond_cty[cty]['diff_prem_95'], linestyle='dotted', label=f'Country {cty} - saved prem.', color=color)
    ax_2.tick_params(axis='y', labelcolor=color)

    lines1, labels1 = ax.get_legend_handles_labels()  
    lines2, labels2 = ax_2.get_legend_handles_labels()  

    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.show()


def plot_coverag(countries, prot_share_arr, metrics_sng_dic, coverage_pool_dic, MES_cty_dic, es_metrics_pool, premium_dic_pool, nominal_arr_pool, prc_mth='ibrd'):

    coverage = {
        country: {
            'cov_sng': [],
            'cov_mlt': [],
            'required': [],
            'abs_prem_95': [],
            'abs_prem_99': []
        }
        for country in countries
    }


    for cty in countries:
        for i in range(len(prot_share_arr)):
            ps_str = str(round(prot_share_arr[i], 2))

            coverage[cty]['cov_sng'].append(np.mean(metrics_sng_dic[ps_str]['Coverage']))
            coverage[cty]['cov_mlt'].append(np.mean(coverage_pool_dic[ps_str][cty]['coverage']))

            coverage[cty]['abs_prem_95'].append(MES_cty_dic[ps_str][cty]['95'] / es_metrics_pool[ps_str]['ES_95_ann'] * premium_dic_pool[ps_str][prc_mth] * nominal_arr_pool[i])
            coverage[cty]['abs_prem_99'].append(MES_cty_dic[ps_str][cty]['99'] / es_metrics_pool[ps_str]['ES_99_ann'] * premium_dic_pool[ps_str][prc_mth] * nominal_arr_pool[i])



    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 6))

    for cty in countries:
        ax1.plot(prot_share_arr, coverage[cty]['cov_sng'], label=f'Country {cty} - single')
        ax1.plot(prot_share_arr, coverage[cty]['cov_mlt'], label=f'Country {cty} - pooled')
    ax1.set_xlabel('Nominal [share of GDP]')
    ax1.set_ylabel('Coverage []')
    ax1.set_title('Coverage vs. Nominal')
    ax1.legend()

    num_countries = len(coverage.keys())
    countries = list(coverage.keys())  # Convert keys to a list

    # Define the number of rows and columns for the subplots
    n_cols = 4  # Choose the number of columns (for a wide layout)
    n_rows = math.ceil(num_countries+1 / n_cols)  # Determine the number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 6 * n_rows))

    # Flatten the axes array in case of more than 1 row
    if n_rows > 1:
        axes = axes.flatten()

    for i, cty in enumerate(countries):
        ax2.plot(abs_prem_ibrd_174, coverage[cty]['cov_sng'], label='Country 1 - single')
        ax2.plot(coverage[cty]['abs_prem_95'], coverage[cty]['cov_mlt'], label='Country 1 - pooled - ES95')
        ax2.plot(coverage[cty]['abs_prem_99'], coverage[cty]['cov_mlt'], label='Country 1 - pooled - ES99')
        ax2.set_xlabel('Premium [USD]')
        ax2.set_ylabel('Coverage []')
        ax2.set_title(f'Country {countries[i]} - Coverage vs Premiums')
        ax2.legend()


    plt.show()