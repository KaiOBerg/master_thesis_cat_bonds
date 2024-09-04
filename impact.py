import pandas as pd

#import climada stuff
from climada.entity.impact_funcs import trop_cyclone
from climada.engine import ImpactCalc


def init_imp(exp, haz, grid):
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
    frequ_curve.plot()
    #save impact per exposure point
    imp_per_exp = imp.imp_mat

    #Perform a spatial join to associate each exposure point with calculated impact with a grid cell
    exp_to_grid = exp.gdf.sjoin(grid, how='left', predicate="within")

    #group each exposure point according to grid cell letter
    agg_exp = exp_to_grid.groupby('grid_letter').apply(lambda x: x.index.tolist())

    #Dictionary to store the impacts for each grid cell
    imp_grid_csr = {}

    #Loop through each grid cell and its corresponding line numbers
    for letter, line_numbers in agg_exp.items():
        selected_values = imp_per_exp[:, line_numbers] #Select all impact values per grid cell
        imp_grid_csr[letter] = selected_values #Store them in dictionary per grid cell

    imp_grid_evt = {} #total damage for each event per grid cell

    #sum all impacts per grid cell
    for i in imp_grid_csr:
        imp_grid_evt[i] = imp_grid_csr[i].sum(axis=1) #calculate sum of impacts per frid cell
        imp_grid_evt[i] = [matrix.item() for matrix in imp_grid_evt[i]] #single values per event are stored in 1:1 matrix -> only save value

    #transform matrix to data frame
    imp_grid_evt = pd.DataFrame.from_dict(imp_grid_evt)

    return imp, imp_per_exp, agg_exp, imp_grid_evt

