'''Conatains a variety of functions used for analysis and plotting.'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as gridspec


import calc_premium as cp
import prem_ibrd as prib
import simulate_multi_cty_bond as smcb
import alt_pay_opt as apo
import impact as cimp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
def create_tranches(rp_array, tot_losses, ann_losses_alt, ibrd_path, prem_corr=0, peak_mulit=0):
    rows = []
    tranch_df = pd.DataFrame(columns=['RP', 'Loss'])
    for i in rp_array:
        loss = calc_rp_bnd_lss(tot_losses, i)
        rows.append({'RP': i, 'Loss': (loss)})
    rows.append({'RP': 'Max', 'Loss': (calc_rp_bnd_lss(tot_losses, len(tot_losses)+1))})

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

    finance_scheme = pd.read_excel(OUTPUT_DIR.joinpath("Data/fs_high_inc.xlsx"))
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


def plt_cty_level_ps_pool(bond_metrics_sng_dic, premium_dic_sng_dic, sng_ann_ret_df_ibrd, countries, prem_diff, scenarios_dic, nominal_dic, x_axis, save_path):
    #define list of used pricing methods
    pricing_list = ['IBRD-Pricing', 'Chatoro-Pricing', 'Benchmark-Pricing']

    # get all scenarios and nominal values from dictionaries
    scenario_1 = scenarios_dic['s1']
    scenario_3 = scenarios_dic['s3']
    scenario_5 = scenarios_dic['s5']

    nominal_s1 = nominal_dic['s1']
    nominal_s3 = nominal_dic['s3']
    nominal_s5 = nominal_dic['s5']

    for idx, nominal in enumerate(nominal_dic.values()):
        name = f'nominal_{idx+1}'
        globals()[name] = nominal  # Assign the nominal value to a variable with the name


    if x_axis == 'ed250_rel':

        x_labels = [
            50.9, 37.8, 36.4, 35.5, 35.4, 35.3, 35.1, 34.8, 34.0, 34.0,
            33.7, 31.5, 29.6, 28.2, 25.7, 25.5, 24.6, 24.3, 22.0, 17.7,
            14.4, 12.7, 11.7, 7.4, 6.1
        ]

        mauritius_position_mono = (-20, 25)
        mauritius_position_tric = (-20, 25)
        mauritius_position_pent = (-20, 25)
        mauritius_position_attr = (-20, 25)
        marshall_position_mono = (-15, -40)
        marshall_position_tric = (40, -15)
        marshall_position_pent = (40, -45)
        marshall_position_attr = (0, 95)
        jamaica_position_mono = (-10, 30)
        jamaica_position_tric = (40, 0)
        jamaica_position_pent = (-70, -5)
        jamaica_position_attr = (40, 0)

    elif x_axis == 'el':
        x_labels = []
        for key in premium_dic_sng_dic:
            x_labels.append(premium_dic_sng_dic[key]['exp_loss']*100)

        mauritius_position_mono = (-20, 20)
        mauritius_position_tric = (-20, 15)
        mauritius_position_pent = (-20, 20)
        mauritius_position_attr = (-20, 20)
        marshall_position_mono = (-30, -40)
        marshall_position_tric = (-20, -20)
        marshall_position_pent = (-20, -20)
        marshall_position_attr = (-38, 55)
        jamaica_position_mono = (-20, 20)
        jamaica_position_tric = (-20, 35)
        jamaica_position_pent = (-20, -20)
        jamaica_position_attr = (-20, -20)

    elif x_axis == 'ed250_abs':
        gdp_per_cty = [11.41, 0.50, 0.87, 14.5, 0.87, 1.41, 13.83, 4.79, 1.5, 0.88,
                       1.04, 78.87, 9.95, 0.91, 4.43, 20.79, 107.49, 0.01, 2.05, 0.49,
                       1.53, 1.23, 0.24, 0.24, 0.26]
        ed_250_rel = [
            50.9, 37.8, 36.4, 35.5, 35.4, 35.3, 35.1, 34.8, 34.0, 34.0,
            33.7, 31.5, 29.6, 28.2, 25.7, 25.5, 24.6, 24.3, 22.0, 17.7,
            14.4, 12.7, 11.7, 7.4, 6.1
        ]

        x_labels = [gdp *  10e3 * (ed / 100) for gdp, ed in zip(gdp_per_cty, ed_250_rel)]

        mauritius_position_mono = (-20, -20)
        mauritius_position_tric = (-20, -20)
        mauritius_position_pent = (20, -20)
        mauritius_position_attr = (20, 0)
        marshall_position_mono = (-80, -20)
        marshall_position_tric = (-20, -20)
        marshall_position_pent = (-90, -18)
        marshall_position_attr = (-90, -18)
        jamaica_position_mono = (40, 0)
        jamaica_position_tric = (40, 0)
        jamaica_position_pent = (-40, -20)
        jamaica_position_attr = (-40, -20)

    else:
        raise ValueError("Invalid x_axis value. Use 'ed250_rel', 'ed250_abs, or 'el'.")



    for p_name in pricing_list:
        if p_name == 'IBRD-Pricing':
            prem_diff_label = 'ibrd'
            if x_axis == 'ed250_abs':
                jamaica_position_attr = (-40, -10)
                marshall_position_attr = (-90, -10)
        elif p_name == 'Chatoro-Pricing':
            prem_diff_label = 'regression'
            if x_axis == 'ed250_abs':
                jamaica_position_attr = (-40, -20)
                marshall_position_attr = (-90, -18)
        elif p_name == 'Benchmark-Pricing':
            prem_diff_label = 'required'
        fig = plt.figure(figsize=(12,9))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 0.33, 1, 0.33])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        ax5 = fig.add_subplot(gs[1, 0])
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[3, 0])
        ax8 = fig.add_subplot(gs[3, 1])

        colors = ['#0072B2',  # Blue
                  '#D55E00',  # Vermilion (reddish-orange)
                  '#F0E442',  # Yellow
                  '#009E73',  # Bluish green
                  '#CC79A7']  # Reddish purple / Magenta
        
        

        for p in range(len(scenario_1)):
            indices = [countries.index(x) for x in scenario_1[p]]
            prem_diff_pool_ibrd = [prem_diff[prem_diff_label]['P1'][i] for i in indices]
            x_labels_pool = [x_labels[i] for i in indices]

            ax1.scatter(x_labels_pool, np.array(prem_diff_pool_ibrd)*100, color=colors[p], alpha=0.5, marker='o', zorder=2)
        ax1.hlines(
            prem_diff[prem_diff_label]['P1'][-1]*100,
            5,
            55000000000,
            colors='red',
            linestyles='solid',
            linewidth=2,
            zorder=1  # Lower zorder puts the line in the background
        )

        # Annotate Mauritius
        ax1.annotate(
            'Mauritius',
            (x_labels[0], prem_diff[prem_diff_label]['P1'][0]*100),
            textcoords="offset points",
            xytext=mauritius_position_mono,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        # Annotate Marhsall Islands
        ax1.annotate(
            'Marhsall Islands',
            (x_labels[23], prem_diff[prem_diff_label]['P1'][23]*100),
            textcoords="offset points",
            xytext=marshall_position_mono,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        # Annotate Tonga
        ax1.annotate(
            'Jamaica',
            (x_labels[6], prem_diff[prem_diff_label]['P1'][6]*100),
            textcoords="offset points",
            xytext=jamaica_position_mono,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        ax1.set_title('MonoCAT-Portfolio', fontsize=12, fontweight='bold')

        for p in range(len(scenario_3)):
            indices = [countries.index(x) for x in scenario_3[p]]
            prem_diff_pool_ibrd = [prem_diff[prem_diff_label]['P3'][i] for i in indices]
            x_labels_pool = [x_labels[i] for i in indices]

            ax2.scatter(x_labels_pool, np.array(prem_diff_pool_ibrd)*100, color=colors[p], alpha=0.5, marker='o', zorder=2)
        ax2.hlines(
            prem_diff[prem_diff_label]['P3'][-1]*100,
            5,
            55000000000,
            colors='red',
            linestyles='solid',
            linewidth=2,
            zorder=1  # Lower zorder puts the line in the background
        )

        # Annotate Mauritius
        ax2.annotate(
            'Mauritius',
            (x_labels[0], prem_diff[prem_diff_label]['P3'][0]*100),
            textcoords="offset points",
            xytext=mauritius_position_tric,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        # Annotate Marhsall Islands
        ax2.annotate(
            'Marhsall Islands',
            (x_labels[23], prem_diff[prem_diff_label]['P3'][23]*100),
            textcoords="offset points",
            xytext=marshall_position_tric,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        # Annotate Tonga
        ax2.annotate(
            'Jamaica',
            (x_labels[6], prem_diff[prem_diff_label]['P3'][6]*100),
            textcoords="offset points",
            xytext=jamaica_position_tric,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        ax2.set_title('TriCAT-Portfolio', fontsize=12, fontweight='bold')

        for p in range(len(scenario_5)):
            indices = [countries.index(x) for x in scenario_5[p]]
            prem_diff_pool_ibrd = [prem_diff[prem_diff_label]['P5'][i] for i in indices]
            x_labels_pool = [x_labels[i] for i in indices]

            ax3.scatter(x_labels_pool, np.array(prem_diff_pool_ibrd)*100, alpha=0.5, color=colors[p], marker='o', zorder=2)
        ax3.hlines(
            prem_diff[prem_diff_label]['P5'][-1]*100,
            5,
            55000000000,
            colors='red',
            linestyles='solid',
            linewidth=2,
            zorder=1  # Lower zorder puts the line in the background
        )

        # Annotate Mauritius
        ax3.annotate(
            'Mauritius',
            (x_labels[0], prem_diff[prem_diff_label]['P5'][0]*100),
            textcoords="offset points",
            xytext=mauritius_position_pent,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        # Annotate Marhsall Islands
        ax3.annotate(
            'Marhsall Islands',
            (x_labels[23], prem_diff[prem_diff_label]['P5'][23]*100),
            textcoords="offset points",
            xytext=marshall_position_pent,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        # Annotate Tonga
        ax3.annotate(
            'Jamaica',
            (x_labels[6], prem_diff[prem_diff_label]['P5'][6]*100),
            textcoords="offset points",
            xytext=jamaica_position_pent,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        ax3.set_title('PentaCAT-Portfolio', fontsize=12, fontweight='bold')


        for p in range(len(scenario_5)):
            indices = [countries.index(x) for x in scenario_5[p]]
            prem_diff_pool_ibrd = [prem_diff[prem_diff_label]['P5A'][i] for i in indices]
            x_labels_pool = [x_labels[i] for i in indices]

            ax4.scatter(x_labels_pool, np.array(prem_diff_pool_ibrd)*100, alpha=0.5, color=colors[p], marker='o', zorder=2)
        ax4.hlines(
            prem_diff[prem_diff_label]['P5A'][-1]*100,
            5,
            55000000000,
            colors='red',
            linestyles='solid',
            linewidth=2,
            zorder=1  # Lower zorder puts the line in the background
        )


        # Annotate Premium savings
        #ax4.text(
        #    20, 
        #    78,
        #    f'Premium Saving: {prem_diff[prem_diff_label]["P5A"][-1]*100:.0f}%',
        #    fontsize=12,
        #    color='black',
        #)

        # Annotate Mauritius
        ax4.annotate(
            'Mauritius',
            (x_labels[0], prem_diff[prem_diff_label]['P5A'][0]*100),
            textcoords="offset points",
            xytext=mauritius_position_attr,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        # Annotate Marhsall Islands
        ax4.annotate(
            'Marhsall Islands',
            (x_labels[23], prem_diff[prem_diff_label]['P5A'][23]*100),
            textcoords="offset points",
            xytext=marshall_position_attr,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        # Annotate Tonga
        ax4.annotate(
            'Jamaica',
            (x_labels[6], prem_diff[prem_diff_label]['P5A'][6]*100),
            textcoords="offset points",
            xytext=jamaica_position_attr,
            ha='left',
            fontsize=10,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='black', lw=1)
        )

        ax4.set_title('Premium-Adjusted-PentaCAT-Portfolio', fontsize=12, fontweight='bold')

        # Add subplot labels in the top left of each plot
        ax1.text(-0.05, 1.05, 'a)', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        ax2.text(-0.05, 1.05, 'b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        ax3.text(-0.05, 1.05, 'c)', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        ax4.text(-0.05, 1.05, 'd)', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_ylim(-0.22*100, 0.89*100)
            ax.set_ylabel('Premium Saving [%]', fontsize=12)
            ax.hlines(0, 5, 55000000000, colors='black', linestyles='dashed', linewidth=1)
            if x_axis == 'ed250_rel':
                ax.set_xlim(5, 55)
                ax.set_xlabel(r"$E[D_{250}]$/GDP [%]", fontsize=12)
            elif x_axis == 'el':
                ax.set_xlim(0, 4)
                ax.set_xlabel('Expected Loss [%]', fontsize=12)
            elif x_axis == 'ed250_abs':
                ax.set_xlim(10, max(x_labels) + 500000)  # Set x-axis limits to a reasonable range
                ax.set_xscale('log')
                ax.set_xlabel('$E[D_{250}]$ [mUSD]', fontsize=12)
            ax.set_axisbelow(True)
            ax.grid(True)


        height_bar = 0.8
        axbarh_y = ['Pool 1', 'Pool 2', 'Pool 3', 'Pool 4', 'Pool 5']
        ax5.barh(axbarh_y,
             [nominal_s1['1']/1000000, 0.0, 0.0, 0.0, 0.0],
             height=height_bar,
             color=colors[0],
             linewidth=1)
        ax5.set_yticklabels(['Pool 1', '', '', '', ''])

        iterator = 0
        for key in nominal_s3:
            ax6.barh(f'Pool {key}', nominal_s3[key]/1000000, height_bar, color=colors[iterator], linewidth=1)
            iterator += 1
        ax6.barh('Pool 4', 0.0, height_bar, color=colors[3], linewidth=1)
        ax6.barh('Pool 5', 0.0, height_bar, color=colors[4], linewidth=1)
        ax6.set_yticklabels(['Pool 1', 'Pool 2', 'Pool 3', '', ''])
        iterator = 0
        for key in nominal_s5:
            ax7.barh(f'Pool {key}', nominal_s5[key]/1000000, height_bar, color=colors[iterator], linewidth=1)
            iterator += 1
        iterator = 0
        for key in nominal_s5:
            ax8.barh(f'Pool {key}', nominal_s5[key]/1000000, height_bar, color=colors[iterator], linewidth=1)
            iterator += 1

        # Apply log scale to x-axes
        for ax in [ax5, ax6, ax7, ax8]:

            ax.set_xscale('log')
            ax.set_xlim(10,60000000000/1000000)  # Set x-axis limits to a reasonable range
            ax.set_xlabel('Principal [mUSD]', fontsize=12)

        fig.tight_layout()
        fig.savefig(f'{save_path}/Plots/cty_level_ps_{x_axis}_{p_name}.pdf', bbox_inches='tight')
        plt.show()


def get_prem_diff(premium_dic_pool, bond_metrics_sng_dic, nominal_dic, sng_ann_ret, countries):
    # use returns, premium payments and payouts to derive efficient frontiers, premium savings and insurance multiples for each scenario and pricing approach
    premiums_abs_keys = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P6A', 'P6AB']
    premiums_abs = {key: {} for key in premiums_abs_keys}
    premium_methods = ['ibrd','required', 'regression'] 
    premium_methods_title = ['IBRD-Pricing', 'Benchmark-Pricing', 'Chatoro-Pricing'] 

    for idx, nominal in enumerate(nominal_dic.values()):
        name = f'nominal_{idx+1}'
        globals()[name] = nominal  # Assign the nominal value to a variable with the name
    for idx, premium in enumerate(premium_dic_pool.values()):
        name = f'premium_{idx+1}'
        globals()[name] = premium  # Assign the premium value to a variable with the name

    prem_diff = {'ibrd': {},'required': {}, 'regression': {}}

    for j, prem_mode in enumerate(premium_methods): 
        premiums_pool_s1 = {}
        premiums_pool_s2 = {}
        premiums_pool_s3 = {}
        premiums_pool_s4 = {}
        premiums_pool_s5 = {}
        premiums_pool_s5_a = {}
        premiums_pool_s5_a_b = {}
        premiums_pool_s5_a_r = {}
        for pool, prem_modes in premiums_s1.items():
            if prem_mode in prem_modes:
                for key, values in prem_modes[prem_mode].items():
                    values = np.array(values)
                    if key in premiums_pool_s1: 
                        premiums_pool_s1[key] += np.sum(values * nominal_s1[pool])
                    else:  
                        premiums_pool_s1[key] = np.sum(values * nominal_s1[pool])
        for pool, prem_modes in premiums_s2.items():
            if prem_mode in prem_modes:
                for key, values in prem_modes[prem_mode].items():
                    values = np.array(values)
                    if key in premiums_pool_s2:  
                        premiums_pool_s2[key] += np.sum(values * nominal_s2[pool])
                    else:  # Otherwise, initialize it with the current value
                        premiums_pool_s2[key] = np.sum(values * nominal_s2[pool])
        for pool, prem_modes in premiums_s3.items():
            if prem_mode in prem_modes:
                for key, values in prem_modes[prem_mode].items():
                    values = np.array(values)
                    if key in premiums_pool_s3:  
                        premiums_pool_s3[key] += np.sum(values * nominal_s3[pool])
                    else:  # Otherwise, initialize it with the current value
                        premiums_pool_s3[key] = np.sum(values * nominal_s3[pool])
        for pool, prem_modes in premiums_s4.items():
            if prem_mode in prem_modes:
                for key, values in prem_modes[prem_mode].items():
                    values = np.array(values)
                    if key in premiums_pool_s4:  
                        premiums_pool_s4[key] += np.sum(values * nominal_s4[pool])
                    else:  # Otherwise, initialize it with the current value
                        premiums_pool_s4[key] = np.sum(values * nominal_s4[pool])
        for pool, prem_modes in premiums_s5.items():
            if prem_mode in prem_modes:
                for key, values in prem_modes[prem_mode].items():
                    values = np.array(values)
                    if key in premiums_pool_s5: 
                        premiums_pool_s5[key] += np.sum(values * nominal_s5[pool])
                    else:  # Otherwise, initialize it with the current value
                        premiums_pool_s5[key] = np.sum(values * nominal_s5[pool])
        for pool, prem_modes in premiums_s5_a.items():
            if prem_mode in prem_modes:
                for key, values in prem_modes[prem_mode].items():
                    values = np.array(values)
                    if key in premiums_pool_s5_a:  
                        premiums_pool_s5_a[key] += np.sum(values * nominal_s5_a[pool])
                    else:  # Otherwise, initialize it with the current value
                        premiums_pool_s5_a[key] = np.sum(values * nominal_s5_a[pool])
        for pool, prem_modes in premiums_s5_a_b.items():
            if prem_mode in prem_modes:
                for key, values in prem_modes[prem_mode].items():
                    values = np.array(values)
                    if key in premiums_pool_s5_a_b:  
                        premiums_pool_s5_a_b[key] += np.sum(values * nominal_s5_a_b[pool])
                    else:  # Otherwise, initialize it with the current value
                        premiums_pool_s5_a_b[key] = np.sum(values * nominal_s5_a_b[pool])
        for pool, prem_modes in premiums_s5_a_r.items():
            if prem_mode in prem_modes:
                for key, values in prem_modes[prem_mode].items():
                    values = np.array(values)
                    if key in premiums_pool_s5_a_r: 
                        premiums_pool_s5_a_r[key] += np.sum(values * nominal_s5_a_r[pool])
                    else:  # Otherwise, initialize it with the current value
                        premiums_pool_s5_a_r[key] = np.sum(values * nominal_s5_a_r[pool])

        sng_cty_premium = []  
        s = {'P0': None, 'P1': [], 'P2': [], 'P3': [], 'P4': [], 'P5': [], 'P5A': []}
        s['P0'] = sng_cty_premium

        if prem_mode == 'ibrd':
            for cty in bond_metrics_sng_dic:
                sng_cty_premium.append(bond_metrics_sng_dic[cty]['Total Premiums'][0]/len(sng_ann_ret['212']))

            s['P0'] = sng_cty_premium
            for cty in countries:
                s['P1'].append(np.sum(premiums_pool_s1[cty])/len(sng_ann_ret['212']))
                s['P2'].append(np.sum(premiums_pool_s2[cty])/len(sng_ann_ret['212']))
                s['P3'].append(np.sum(premiums_pool_s3[cty])/len(sng_ann_ret['212']))
                s['P4'].append(np.sum(premiums_pool_s4[cty])/len(sng_ann_ret['212']))
                s['P5'].append(np.sum(premiums_pool_s5[cty])/len(sng_ann_ret['212']))
                s['P5A'].append(np.sum(premiums_pool_s5_a[cty])/len(sng_ann_ret['212']))

        elif prem_mode == 'regression':
            for cty in bond_metrics_sng_dic:
                sng_cty_premium.append(bond_metrics_sng_dic[cty]['Total Premiums'][1]/len(sng_ann_ret['212']))

            s['P0'] = sng_cty_premium
            for cty in countries:
                s['P1'].append(np.sum(premiums_pool_s1[cty])/len(sng_ann_ret['212']))
                s['P2'].append(np.sum(premiums_pool_s2[cty])/len(sng_ann_ret['212']))
                s['P3'].append(np.sum(premiums_pool_s3[cty])/len(sng_ann_ret['212']))
                s['P4'].append(np.sum(premiums_pool_s4[cty])/len(sng_ann_ret['212']))
                s['P5'].append(np.sum(premiums_pool_s5[cty])/len(sng_ann_ret['212']))
                s['P5A'].append(np.sum(premiums_pool_s5_a_r[cty])/len(sng_ann_ret['212']))

        elif prem_mode == 'required':
            for cty in bond_metrics_sng_dic:
                sng_cty_premium.append(bond_metrics_sng_dic[cty]['Total Premiums'][2]/len(sng_ann_ret['212']))

            s['P0'] = sng_cty_premium
            for cty in countries:
                s['P1'].append(np.sum(premiums_pool_s1[cty])/len(sng_ann_ret['212']))
                s['P2'].append(np.sum(premiums_pool_s2[cty])/len(sng_ann_ret['212']))
                s['P3'].append(np.sum(premiums_pool_s3[cty])/len(sng_ann_ret['212']))
                s['P4'].append(np.sum(premiums_pool_s4[cty])/len(sng_ann_ret['212']))
                s['P5'].append(np.sum(premiums_pool_s5[cty])/len(sng_ann_ret['212']))
                s['P5A'].append(np.sum(premiums_pool_s5_a_b[cty])/len(sng_ann_ret['212']))

        else:
            print('Wrong input premium mode')
            continue

        for key in s:
            prem_diff[prem_mode][key] = (1-(np.array(s[key])/np.array(s['P0']))).tolist()

    return prem_diff

