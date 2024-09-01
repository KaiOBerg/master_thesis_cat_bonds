#import climada stuff
from climada.entity.impact_funcs import trop_cyclone
from climada.engine import ImpactCalc


def init_imp(exp, haz):
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

    imp = ImpactCalc(exp, impf_set, haz).impact(save_mat=True)

    #compute exceedance frequency curve
    frequ_curve = imp.calc_freq_curve()
    frequ_curve.plot()
    imp_per_exp = imp.imp_mat

    return imp, imp_per_exp