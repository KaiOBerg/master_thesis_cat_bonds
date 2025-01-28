'''Conatains a variety of functions used for analysis and plotting.'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import calc_premium as cp
import prem_ibrd as prib
import simulate_multi_cty_bond as smcb
import alt_pay_opt as apo
import impact as cimp

#checks if it is a scaler value
def check_scalar(variable):
    if np.isscalar(variable):
        cor_var = np.array([variable])
    else:
        cor_var = variable
    
    return cor_var

#extract values from dictionary for better display
def get_all_values(d):
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.extend(get_all_values(value))
        else:
            values.append(value)
    return values

#print progress bar when doing a for loop
def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    :param iteration: Current iteration (int)
    :param total: Total iterations (int)
    :param prefix: Prefix string (str)
    :param suffix: Suffix string (str)
    :param decimals: Positive number of decimals in percent complete (int)
    :param length: Character length of bar (int)
    :param fill: Bar fill character (str)
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
    # Print New Line on Complete
    if iteration == total: 
        print()

#show results
def print_sng_bnd_rel_metr(bond_metrics, returns, premium_dic, nominal):
    print('Expected Loss:',round(premium_dic['exp_loss']*100, 1),'%')
    print('Attachment Probability:',round(premium_dic['att_prob']*100,1),'%')
    print('Coverage:',round(bond_metrics['Coverage'][0]*100,1),'%')
    print('Premium Ibrd:',round(premium_dic['ibrd']*100,1),'%;',round(premium_dic['ibrd']*nominal, 0),'USD' )
    print('Premium Chatoro et al.',round(premium_dic['regression']*100,1),'%;',round(premium_dic['regression']*nominal, 0),'USD')
    print('Premium Target Sharpe Ratio',round(premium_dic['required']*100,1),'%;',round(premium_dic['required']*nominal, 0),'USD')
    print('Standard Deviation of Returns',round(np.std(returns['Annual'][0]),2))

#show results
def print_mlt_bnd_rel_metr(countries, returns, premium_dic, tot_coverage_cty, nominal):
    print('Expected Loss: ',round(premium_dic['exp_loss']*100, 1),'%')
    print('Attachment Probability: ',round(premium_dic['att_prob']*100,1),'%')
    for i in countries:
        print(f'Coverage {i}:',round(tot_coverage_cty[i]['coverage']*100,1),'%')
    print('Premium Ibrd: ',round(premium_dic['ibrd']*100,1),'%; ',round(premium_dic['ibrd']*nominal, 0),'USD')
    print('Premium Chatoro et al.',round(premium_dic['regression']*100,1),'%; ',round(premium_dic['regression']*nominal, 0),'USD')
    print('Premium Target Sharpe Ratio',round(premium_dic['required']*100,1),'%; ',round(premium_dic['required']*nominal, 0),'USD')
    print('Standard Deviation Returns',np.std(returns['Annual'][0]))

#calculate return period of losses
def calc_rp_bnd_lss(ann_losses, return_period):
    """
    Compute impacts/payouts for specific return periods using a DataFrame.

    Parameters
    ----------
    df : pandas.Series
        A Series containing annual loss values.
    return_periods : Object
        The return period where we want to compute the exceedance impact/pay.

    Returns
    -------
    A number.
    """

    #annual_losses = ann_losses['losses'].apply(sum)
    annual_losses = ann_losses
    df = pd.DataFrame(annual_losses.sort_values(ascending=True), columns=['losses'])
    df['Rank'] = df.rank(method='first', ascending=False)
    df['RP'] = (len(df) + 1)/df['Rank']
    df = df.sort_values(by='RP')
    sorted_rp = df['RP'].values
    sorted_losses = df['losses'].values
    calc_value = np.interp(return_period, sorted_rp, sorted_losses)

    return calc_value

#create tranches for multi country bond
def create_tranches(rp_array, ann_losses, ann_losses_alt, ibrd_path, prem_corr=0, peak_mulit=0):
    rows = []
    tranch_df = pd.DataFrame(columns=['RP', 'Loss'])
    for i in rp_array:
        loss = calc_rp_bnd_lss(ann_losses, i)
        rows.append({'RP': i, 'Loss': (loss)})
    rows.append({'RP': 'Max', 'Loss': (calc_rp_bnd_lss(ann_losses, len(ann_losses)))})

    # Combine the rows into a DataFrame
    tranches = pd.concat([tranch_df, pd.DataFrame(rows)], ignore_index=True)
    tranches['nominal'] = 0.0
    tranches['nominal'] = tranches['Loss'].diff()
    tranches.at[0, 'nominal'] = tranches.at[0, 'Loss']
    el = 0
    tranches['expected_loss'] = 0.0
    tranches['expected_loss_own'] = 0.0
    tranches['lower_bound'] = 0.0
    tranches['upper_bound'] = 0.0
    tranches['premium_ibrd'] = 0.0
    tranches['premium_regression'] = 0.0
    tranches['premium_required'] = 0.0
    tranche_losses_dic = {}

    # Calculate lower and upper bounds, and expected loss
    for i in tranches.index:
        # Determine layer boundaries
        tranches.at[i, 'lower_bound'] = 0 if i == 0 else tranches.at[i - 1, 'Loss']
        tranches.at[i, 'upper_bound'] = tranches.at[i, 'Loss']

        # Losses within the tranche layer
        annual_losses = ann_losses_alt['losses'].apply(sum)
        sum_term_loss = 0
        tranche_loss_term = 0
        tranche_losses = []
        for j in range(len(annual_losses)):
            tranche_losses_tmp = (
                np.clip(annual_losses[j] + sum_term_loss, tranches.at[i, 'lower_bound'], tranches.at[i, 'upper_bound']) 
                - tranches.at[i, 'lower_bound']
            )
            tranche_losses_tmp -= tranche_loss_term
            tranche_loss_term += tranche_losses_tmp
            tranche_losses.append(tranche_losses_tmp)
            sum_term_loss += annual_losses[j]
            if (j + 1) % 3 == 0:
                sum_term_loss = 0
                tranche_loss_term = 0
        tranche_losses = np.array(tranche_losses)
        tranche_losses_dic[i] = tranche_losses

        # Expected loss for the tranche
        tranche_el = np.mean(tranche_losses)
        el += tranche_el
        tranches.at[i, 'expected_loss'] = tranche_el

        if tranches.at[i, 'nominal'] > 0:
            tranches_loss_own =  np.array(tranche_losses)/tranches.at[i, 'nominal']
        else:
            tranches_loss_own = 0
        tranches.at[i, 'expected_loss_own'] = np.mean(tranches_loss_own)
        
    # Calculate share of expected loss
    tranches['share_expected_loss'] = tranches['expected_loss'] / el

    tranches['nominal'] = tranches['Loss'].diff()
    tranches.at[0, 'nominal'] = tranches.at[0, 'Loss']

    params_ibrd = prib.init_prem_ibrd(file_path=ibrd_path, want_plot=False)
    a, k, b = params_ibrd

    tranches['premium_ibrd'] = prib.monoExp(tranches['expected_loss_own']*100, a, k, b) * tranches['expected_loss_own'] + prem_corr
    tranches['premium_regression'] = cp.calc_premium_regression(tranches['expected_loss_own'] *100, peak_mulit)/100 + prem_corr
    for i in tranches.index:
        tranches.at[i, 'premium_required'] = smcb.init_prem_sharpe_ratio_tranches(ann_losses_alt, tranches.at[i, 'nominal'], tranche_losses_dic[i], 0.0, 0.5) + prem_corr

    return tranches

#plot histogram of tc events according to tc category
def plot_TC_hist(tc, categories, ax):
    #should I limit it to on land centroids?
    dense_intensity = np.squeeze(np.asarray(tc.intensity.todense()).flatten())
    
    ax.hist(dense_intensity, bins=categories[1:], density=True, alpha=0.5, color='k', edgecolor='k', lw=2, zorder=3, label='STORM-P')
    ax.set_ylabel('Tropical cyclone intensity over land (%)', fontsize=12)
    
    locs, xlabels = plt.xticks()
    newlabel = ['  Trop. storm', '   Cat. 1', ' Cat. 2', '   Cat. 3', '    Cat. 4', '    Cat. 5', '']
    ax.set_xlabel('Tropical cyclone category', fontsize=12)
    plt.xticks(categories, newlabel,horizontalalignment='left')
    plt.xlim(10, 80)
    ax.legend(loc="upper left")
    
    return ax


def plot_vulnerability(impf_TC, fun_id, ax):
    ax = plot_impf(impf_TC, fun_id, ax)

    handles, labels = plt.gca().get_legend_handles_labels()
    temp = {k:v for k,v in zip(labels, handles)}    
    ax.legend(temp.values(), temp.keys(), loc="lower right")
    ax.set_ylabel('Damage (%)', fontsize=12)
    plt.ylim(0, 100)

    return ax


def plot_impf(impf_TC, fun_id, ax):
    mdd_impf = impf_TC.get_func(fun_id=fun_id)[0].mdd*100
    intensity_mdd = impf_TC.get_func(fun_id=fun_id)[0].intensity
    ax.plot(intensity_mdd, mdd_impf, label='Impact \nfunction', c='k', zorder=2)
    
    return ax 


def plot_payout_structure(optimized_min, optimized_max, nominal, categories, exp, admin_gdf, ax):
    exp_crs = exp.gdf
    exp_crs = exp_crs.to_crs(admin_gdf.crs)
    #Perform a spatial join to associate each exposure point with calculated impact with a grid cell
    exp_to_admin = exp_crs.sjoin(admin_gdf, how='left', predicate="within")
    #group each exposure point according to grid cell letter
    agg_exp = exp_to_admin.groupby('admin_letter').apply(lambda x: x.index.tolist())
    agg_exp_sum = exp_to_admin.groupby('admin_letter')['value'].sum()
    tot_exp = agg_exp_sum['A']
    TS, cat1, cat2, cat3, cat4, cat5, upper_lim = categories
    steps1 = np.array([0, cat1, optimized_min, cat2, cat3, cat4, cat5, optimized_max, upper_lim])
    
    payouts = []
    for i in steps1:
        df = pd.DataFrame({'Intensity': [i]})
        payouts.append(apo.init_alt_payout(optimized_min, optimized_max, df, nominal, False)/tot_exp*100)
    percentages = np.array(payouts)
    ax.plot(steps1, percentages, label='Payout \nfunction', c='k', ls='--', zorder=2)    
    
    return ax


def plot_vul_his(tc, categories, impf_TC, impf_id):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1 = plot_TC_hist(tc, categories, ax1)
    ax2 = ax1.twinx()
    ax2 = plot_vulnerability(impf_TC, impf_id, ax2)

    return ax1

#plot mean damage and payout for different damage levels
def plot_bin_dam_pay(pay_dam_df, nominal):
    # Calculate damage and pay percentages
    dam_perc = pay_dam_df['damage'] / nominal * 100
    dam_perc = dam_perc.clip(lower=0, upper=100)

    pay_perc = pay_dam_df['pay']/nominal*100
    pay_perc = pay_perc.clip(lower=0, upper=100)

    bins = range(10, 101, 10)  

    binned_dam_data = pd.cut(dam_perc, bins=bins, right=True, include_lowest=True)
    binned_pay_data = pd.cut(pay_perc, bins=bins, right=True, include_lowest=True)

    counts_per_bin_dam = dam_perc.groupby(binned_dam_data, observed=False).size()
    mean_per_bin_dam = dam_perc.groupby(binned_dam_data, observed=False).mean()
    mean_per_bin_pay = pay_perc.groupby(binned_pay_data, observed=False).mean()

    x_labels = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']

    result = pd.DataFrame({
        'bin': x_labels,
        'count_dam': counts_per_bin_dam.values,
        'Mean Damage': mean_per_bin_dam.values,
        'Mean Payout': mean_per_bin_pay.values
    })

    plot_data_melted = pd.melt(
        result,
        id_vars='bin',
        value_vars=['Mean Damage', 'Mean Payout'],
        var_name='type',
        value_name='value'
    )

    fig, ax1 = plt.subplots(figsize=(8, 6))

    sns.swarmplot(data=plot_data_melted, x='bin', y='value', hue='type', palette='Set2', ax=ax1)
    ax1.set_ylabel('Damage/Payout [%]', fontsize=12)
    ax1.set_xlabel('Modeled damage per affected unit and event [%]', fontsize=12)
    ax1.legend(loc='upper center')

    ax2 = ax1.twinx()
    sns.barplot(x=x_labels, y=result['count_dam'], alpha=0.4, color='gray', ax=ax2)
    ax2.set_ylabel('Event Count', fontsize=12)
    ax2.tick_params(axis='y')

    plt.tight_layout()
    plt.show()


#create TC histogram for climate change analysis
def plot_TC_hist_cc(tc_dic, categories, ax):
    tc_sets = ["CRCL", "CMCC", "CNRM", "ECEARTH", "HADGEM"]

    category_shares = {tc_set: [] for tc_set in tc_sets}

    n_tc = {}
    for tc_set in tc_dic:
        tc = tc_dic[tc_set]
        n_tc[tc_set] = len(tc.event_id)
        dense_intensity = np.squeeze(np.asarray(tc.intensity.todense()).flatten())
        hist, _ = np.histogram(dense_intensity, bins=categories, density=False)
        category_shares[tc_set] = hist

    total_per_category = np.sum([np.array(category_shares[tc_set]) for tc_set in tc_dic.keys()], axis=0)
    
    for tc_set in tc_dic:
        category_shares[tc_set] = np.array(category_shares[tc_set]) / total_per_category

    bottom = np.zeros(len(categories)-1) 
    for i, tc_set in enumerate(tc_sets):
        shares = category_shares[tc_set] * 100 
        bars = ax.bar(
            categories[:-1],  
            category_shares[tc_set] * 100, 
            width=np.diff(categories),
            bottom=bottom * 100,
            label=f"{tc_set}\nn={n_tc[tc_set]}",
            color=f"C{i}", 
            edgecolor="k",
            linewidth=0.5,
            align="edge",
        )

        iterator = 0
        for rect, share in zip(bars, shares):
            if share > 0: 
                if iterator % 5 == 0 and iterator > 0:
                    x = rect.get_x() + rect.get_width() / 2 - 22
                else:
                    x = rect.get_x() + rect.get_width() / 2
                y = rect.get_y() + rect.get_height() / 2
                ax.text(
                    x,  
                    y,  
                    f"{share:.1f}%",  
                    ha="center", va="center", fontsize=8, color="white"
                )
            iterator += 1

        bottom += category_shares[tc_set]

    ax.set_ylabel("Share per Category (%)", fontsize=12)
    ax.set_xlabel("Tropical Cyclone Category", fontsize=12)

    newlabel = ['  Trop. storm', '   Cat. 1', ' Cat. 2', '   Cat. 3', '    Cat. 4', '    Cat. 5', '']
    ax.set_xticks(categories, newlabel,horizontalalignment='left')
    ax.set_xlim(18, 90)
    ax.set_ylim(0, 100)

    # Add legend
    ax.legend(loc="lower right", fontsize=10)
    
    return ax

#define function to test the developed CAT bond
def prod_test_ibtrac(tc_storms, exp, admin_gdf, nominal, optimized_step1, optimized_step2, imp_per_event_flt, imp_admin_evt_flt, lower_share):
    imp_hist, imp_per_event_hist, imp_admin_evt_hist = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=False) 

    centrs_to_grid = tc_storms.centroids.gdf.sjoin(admin_gdf, how='left', predicate="within")
    agg_exp = centrs_to_grid.groupby('admin_letter').apply(lambda x: x.index.tolist())
    #Initialize a dictionary to hold the calculated statistics
    int_grid_test = {letter: [None] * len(tc_storms.event_id) for letter in agg_exp.keys()}
    int_grid_test['Storm_ID'] = [None] * len(tc_storms.event_id)
    #Iterate over each event
    for i in range(len(tc_storms.event_id)):
        int_grid_test['Storm_ID'][i] = tc_storms.event_id[i]
        #For each grid cell, calculate the desired statistic
        for letter, line_numbers in agg_exp.items():
            selected_values = tc_storms.intensity[i, line_numbers]
            int_grid_test[letter][i] = selected_values.mean()

    int_grid_test = pd.DataFrame.from_dict(int_grid_test)

    minimum_payout = imp_per_event_flt[imp_per_event_flt > 0].min()
    tot_pay_per_event = []
    b = len(tc_storms.event_id)

    for i in range(len(imp_per_event_hist)):
        payout = []
        for j in range(len(int_grid_test.columns)-1):
            min_trig = optimized_step1[j]
            max_trig = optimized_step2[j]
            grid_hazint = int_grid_test.iloc[i,j]
            max_dam = np.max(imp_admin_evt_flt.iloc[:,j])
            if max_dam < nominal:
                max_pay = max_dam
            else: 
                max_pay = nominal

            if grid_hazint >= max_trig:
                payout.append(max_pay)
            elif grid_hazint <= min_trig:
                payout.append(0)
            else:
                payout.append((grid_hazint - min_trig) / (max_trig - min_trig) * max_pay)

        tot_pay_test = np.sum(payout)
        if tot_pay_test > nominal:
            tot_pay_test = nominal
        elif tot_pay_test < minimum_payout:
            tot_pay_test = 0
        else: 
            pass

        tot_pay_per_event.append(tot_pay_test)

    dam_pay_data_ibtracs = pd.DataFrame({
    'imp': imp_per_event_hist,
    'pay': tot_pay_per_event,
    'STORM_ID': tc_storms.event_name,
    })

    min_pay = lower_share * exp.gdf['value'].sum()

    filtered_data = dam_pay_data_ibtracs[dam_pay_data_ibtracs['imp'] >= min_pay]

    squaredDiffs = np.square(filtered_data['imp'] - filtered_data['pay'])
    squaredDiffsFromMean = np.square(filtered_data['imp'] - np.mean(filtered_data['imp']))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"RMSE: {rSquared}")
    
    return dam_pay_data_ibtracs, rSquared


#define function to calcualte the share of each premium donor in the financial scheme - case study 3
def plot_prem_share(weighter=1, eu=True, threshold_other=1, file_path="C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data"):    
    OUTPUT_DIR = Path(file_path)
    EU = ['AUT', 'BEL', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'PRT', 
      'SVK', 'SVN', 'ESP', 'SWE']

    finance_scheme = pd.read_excel(OUTPUT_DIR.joinpath("fs_high_inc.xlsx"))
    finance_scheme = finance_scheme[finance_scheme['Carbon Footprint per Capita (absolute)'] >= 6.5].copy()
    finance_scheme['Relative Carbon Footprint'] = (finance_scheme['Carbon Footprint per Capita (absolute)'] / finance_scheme['Carbon Footprint per Capita (absolute)'].median())**weighter
    finance_scheme['GDP share'] = finance_scheme['GDP (absolute)'] / finance_scheme['GDP (absolute)'].sum() * 100
    finance_scheme['Pay formula'] = finance_scheme['Relative Carbon Footprint'] * finance_scheme['GDP share']
    finance_scheme['Share of Pay'] = finance_scheme['Pay formula'] / finance_scheme['Pay formula'].sum() * 100
    
    if eu:
        eu_share = finance_scheme[finance_scheme['Code'].isin(EU)]['Share of Pay'].sum()
        eu_gdp_share = finance_scheme[finance_scheme['Code'].isin(EU)]['GDP share'].sum()
        eu_row = pd.DataFrame({'Code': ['EU'],'GDP share': [eu_gdp_share] ,'Share of Pay': [eu_share]})
        finance_scheme_no_eu = finance_scheme[~finance_scheme['Code'].isin(EU)]
        finance_scheme_up = pd.concat([finance_scheme_no_eu, eu_row], ignore_index=True)
    else:
        finance_scheme_up = finance_scheme.copy()
    
    
    filtered_data = finance_scheme_up[finance_scheme_up['Share of Pay'] >= threshold_other]
    filtered_data = filtered_data.sort_values(by='Share of Pay', ascending=False, inplace=False)
    rest_value = finance_scheme_up[finance_scheme_up['Share of Pay'] < threshold_other]['Share of Pay'].sum()
    new_row = {'Code': 'Other', 'Share of Pay': rest_value}
    
    if rest_value > 0:
        filtered_data = pd.concat([filtered_data, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    
    unique_codes = finance_scheme_up['Code'].unique().tolist() + ['Other'] 
    color_palette = sns.color_palette("tab20", len(unique_codes))
    color_map = {code: color for code, color in zip(unique_codes, color_palette)}
    filtered_data['Color'] = filtered_data['Code'].map(color_map)

    plt.figure(figsize=(8, 8))
    plt.pie(
        filtered_data['Share of Pay'],
        colors=filtered_data['Color'],
        labels=filtered_data['Code'],
        autopct='%1.1f%%',
        textprops={'rotation_mode': 'anchor', 'va': 'center', 'rotation': -0}  # Rotate labels
    )
    plt.legend(loc="upper right", ncol=2)
    plt.show()

    return_df = pd.DataFrame({"Country": finance_scheme['Code'], "Share": finance_scheme['Share of Pay']/100, "GDP": finance_scheme['GDP (absolute)'],
                               "CEFC": finance_scheme['Carbon Footprint per Capita (absolute)']})

    return return_df