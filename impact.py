import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#import climada stuff
from climada.entity.impact_funcs import trop_cyclone
from climada.engine import ImpactCalc


def init_imp(exp, haz, imp_admin=None, plot_frequ=True):
    #import regional calibrated impact function for TC
    # prepare impact calcuation - after Samuel Eberenz
    # The iso3n codes need to be consistent with the column “region_id” in the 
    # 1. Init impact functions:
    impact_func_set = trop_cyclone.ImpfSetTropCyclone()
    impf_set = impact_func_set.from_calibrated_regional_ImpfSet()
    impf_set.check()

    # get mapping: country ISO3n per region:
    iso3n_per_region = impf_id_per_region = impact_func_set.get_countries_per_region()[2]
    
    code_regions = {'NA1': 1, 'NA2': 2, 'NI': 3, 'OC': 4, 'SI': 5, 'WP1': 6, \
                    'WP2': 7, 'WP3': 8, 'WP4': 9, 'ROW': 10}

    # match exposure with correspoding impact function
    for calibration_region in impf_id_per_region:
        for country_iso3n in iso3n_per_region[calibration_region]:
            exp.gdf.loc[exp.gdf.region_id== country_iso3n, 'impf_TC'] = code_regions[calibration_region]

    #perform impact calcualtion
    imp = ImpactCalc(exp, impf_set, haz).impact(save_mat=True)

    #compute exceedance frequency curve
    frequ_curve = imp.calc_freq_curve()
    if plot_frequ:
        mask = frequ_curve.return_per < 500
        return_period_flt = frequ_curve.return_per[mask]
        impact_flt = frequ_curve.impact[mask]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 6), gridspec_kw={'width_ratios': [10, 5, 5]})

        ax1.plot(return_period_flt, impact_flt, label='Filtered Data')

        # Add labels and title
        ax1.set_title("Impact Frequency Curve")
        ax1.set_xlabel("Return Period [Years]")
        ax1.set_ylabel("Impact [USD]")

        # Create an inset plot (overview of total data)
        inset_ax1 = inset_axes(ax1, width="30%", height="30%", loc='upper left', borderpad=3.0)  # adjust size and position
        inset_ax1.plot(frequ_curve.return_per, frequ_curve.impact, label='Overview Data')
        inset_ax1.set_xlabel("Return Period [Years]", fontsize=8)
        inset_ax1.set_ylabel("Impact [USD]", fontsize=8)

        ax2.plot(frequ_curve.return_per, frequ_curve.impact)
        ax2.set_xscale('log')
        ax2.set_xlabel('Return Period [Years]')
        ax2.set_ylabel('Impact [USD]')
        ax2.set_title('Impact Frequency Curve - Log')

        ax3.plot(frequ_curve.return_per, frequ_curve.impact)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Return Period [Years]')
        ax3.set_ylabel('Impact [USD]')
        ax3.set_title('Impact Frequency Curve - LogLog')

        # Show both plots
        plt.tight_layout()
        plt.show()

    #save impact per exposure point
    imp_per_event = imp.at_event

    if imp_admin:
        #save impact per exposure point
        imp_per_exp = imp.imp_mat

        #Perform a spatial join to associate each exposure point with calculated impact with a grid cell
        exp_to_admin = exp.gdf.sjoin(imp_admin, how='left', predicate="within")

        #group each exposure point according to grid cell letter
        agg_exp = exp_to_admin.groupby('admin_letter').apply(lambda x: x.index.tolist())

        #Dictionary to store the impacts for each grid cell
        imp_admin_csr = {}

        #Loop through each grid cell and its corresponding line numbers
        for letter, line_numbers in agg_exp.items():
            selected_values = imp_per_exp[:, line_numbers] #Select all impact values per grid cell
            imp_admin_csr[letter] = selected_values #Store them in dictionary per grid cell

        imp_admin_evt = {} #total damage for each event per grid cell

        #sum all impacts per grid cell
        for i in imp_admin_csr:
            imp_admin_evt[i] = imp_admin_csr[i].sum(axis=1) #calculate sum of impacts per grid cell
            imp_admin_evt[i] = [matrix.item() for matrix in imp_admin_evt[i]] #single values per event are stored in 1:1 matrix -> only save value

        #transform matrix to data frame
        imp_admin_evt = pd.DataFrame.from_dict(imp_admin_evt)

    return imp, imp_per_event

