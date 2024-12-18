import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes, rasterize
from rasterio.transform import from_origin, from_bounds
from shapely.geometry import box, shape


bond_cache = {}

def get_bond_metrics(pool, pay_dam_pool_it, nominal_pool_it):
    pool_key = tuple(sorted(pool))  # Create a unique key for the pool
    if pool_key not in bond_cache:
        # If result isn't cached, compute and store it
        pay_dam_temp = {c: pay_dam_pool_it[c] for c in pool}
        nominal_temp = {c: nominal_pool_it[c] for c in pool}
        bond_metrics, returns, tot_coverage, premium_dic, nominal, es_metrics, MES_cty = bond_fct.mlt_cty_bond(
            countries=pool,
            pay_dam_df_dic=pay_dam_temp,
            nominals_dic=nominal_temp,
            opt_cap=True,
        )
        bond_cache[pool_key] = {
            "Returns": returns
            }
    return bond_cache[pool_key]


def init_bond(events_per_year, premium, risk_free_rates, nominal, countries, nominal_dic_cty=None):   
    simulated_ncf = []
    simulated_premium = []
    tot_payout = []
    tot_damage = []
    coverage_cty = {}
    for code in countries:
        coverage_cty[code] = {'payout': 0, 'damage': 0}
    rf_rates_list = []
    metrics = {}    
    cur_nominal = nominal
    cur_nom_cty = nominal_dic_cty.copy() if nominal_dic_cty is not None else {int(country): 1 for country in countries}

        
    for k in range(term):
        rf = check_rf(risk_free_rates, k)
        rf_rates_list.append(rf)
        if events_per_year[k].empty:
            premium_ann = cur_nominal * premium
            net_cash_flow_ann = (cur_nominal * (premium + rf))
            sum_payouts_ann = 0
            sum_damages_ann = 0
        else:
            events_per_year[k] = events_per_year[k].sort_values(by='month')
            net_cash_flow_ann = []
            premium_ann = []
            sum_payouts_ann = []
            sum_damages_ann = []
            months = events_per_year[k]['month'].tolist()
            cties = events_per_year[k]['country_code'].tolist()
            pay = events_per_year[k]['pay'].tolist()
            dam = events_per_year[k]['damage'].tolist()
            ncf_pre_event = (cur_nominal * (premium + rf)) / 12 * months[0]
            net_cash_flow_ann.append(ncf_pre_event)
            premium_ann.append(cur_nominal * premium / 12 * (months[0]))
            cty_payouts_event = {country: [] for country in countries}
            cty_damages_event = {country: [] for country in countries}
            for o in range(len(events_per_year[k])):
                payout = pay[o]
                cty = cties[o]
                damage = dam[o]
                month = months[o]

                if payout == 0 or cur_nominal == 0 or cur_nom_cty[int(cty)] == 0:
                    event_payout = 0
                elif payout > 0:
                    event_payout = payout
                    if nominal_dic_cty is not None:
                        cur_nom_cty[int(cty)] -= event_payout
                        if cur_nom_cty[int(cty)] < 0:
                            event_payout += cur_nom_cty[int(cty)]
                            cur_nom_cty[int(cty)] = 0
                    cur_nominal -= event_payout
                    if cur_nominal < 0:
                        event_payout += cur_nominal
                        cur_nominal = 0
                    else:
                        pass
                if o + 1 < len(events_per_year[k]):
                    nex_month = months[o+1] 
                    premium_post_event = (cur_nominal * premium) / 12 * (nex_month - month)
                    ncf_post_event = ((cur_nominal * (premium + rf)) / 12 * (nex_month - month)) - event_payout
                else:
                    premium_post_event = (cur_nominal * premium) / 12 * (12- month)
                    ncf_post_event = ((cur_nominal * (premium + rf)) / 12 * (12- month)) - event_payout

                net_cash_flow_ann.append(ncf_post_event)
                premium_ann.append(premium_post_event)
                sum_payouts_ann.append(event_payout)
                sum_damages_ann.append(damage)
                cty_payouts_event[cty].append(event_payout)
                cty_damages_event[cty].append(damage)

            for key in cty_payouts_event.keys():
                coverage_cty[key]['payout'] += sum(cty_payouts_event[key])
                coverage_cty[key]['damage'] += sum(cty_damages_event[key])

        tot_payout.append(np.sum(sum_payouts_ann))
        tot_damage.append(np.sum(sum_damages_ann))
        simulated_ncf.append(np.sum(net_cash_flow_ann))
        simulated_premium.append(np.sum(premium_ann))
    simulated_ncf_rel = list(np.array(simulated_ncf) / nominal)
    metrics['tot_payout'] = np.sum(tot_payout)
    metrics['tot_damage'] = np.sum(tot_damage)
    metrics['tot_premium'] = np.sum(simulated_premium)
    if np.sum(tot_payout) == 0:
        tot_pay = np.nan
    else:
        tot_pay = np.sum(tot_payout)
    metrics['tot_pay'] = tot_pay

    return simulated_ncf_rel, metrics, rf_rates_list, coverage_cty



# Define raster properties
pixel_size = 0.008333  # Size of each pixel in degrees (adjust this value as needed)
buffer_size = 0.5  # Buffer size in degrees to expand the raster bounds (adjust this value as needed)
grid_size = 0.3 # Size of each grid cell in degrees (adjust this value as needed)


def init_grid(exp, plot_rst=True):
    minx, miny, maxx, maxy = exp.gdf.total_bounds  # Get bounding box of the GeoDataFrame

    # Expand the bounds by the buffer size
    minx -= buffer_size
    miny -= buffer_size
    maxx += buffer_size
    maxy += buffer_size

    # Calculate the number of rows and columns for the raster
    nrows = int((maxy - miny) / pixel_size) + 1
    ncols = int((maxx - minx) / pixel_size) + 1

    # Define the transformation matrix
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    # Initialize the raster grid with NaNs 
    raster = np.full((nrows, ncols), np.nan)

    # Loop through each point in the GeoDataFrame to assign values to the raster
    for _, row in exp.gdf.iterrows():
        # Extract x (longitude) and y (latitude) from the geometry
        x, y = row.geometry.x, row.geometry.y

        # Calculate the column and row index for each point
        col = int((x - minx) / pixel_size)
        row_idx = int((maxy - y) / pixel_size)

        # Assign the value to the corresponding cell in the raster
        raster[row_idx, col] = row['value']

    # Write the raster to a GeoTIFF file using rasterio
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=1,  # Number of bands
            dtype=rasterio.float32,  # Data type for the raster values
            crs='EPSG:4326',  # Coordinate reference system
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1)  # Write raster data to the first band
            
            # Read back the raster data from the in-memory file
            raster_data = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
    
    # Generate grid cells over the bounding box
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)

    grid_cells = [box(x, y, x + grid_size, y + grid_size) for x in x_coords for y in y_coords]
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    
    # Extract the shapes of the non-zero areas (islands)
    mask = raster_data > 0  # Create a mask for non-zero values
    island_shapes = list(shapes(raster_data, mask=mask, transform=transform))

    # Create a GeoDataFrame of island polygons
    island_polygons = [shape(geom) for geom, value in island_shapes if value > 0]
    islands_gdf = gpd.GeoDataFrame({'geometry': island_polygons}, crs=crs)

    # Select grid cells that intersect with the islands
    intersecting_cells = gpd.sjoin(grid_gdf, islands_gdf, how='inner', predicate='intersects')
    intersecting_cells = intersecting_cells.drop_duplicates(subset='geometry') #remove duplicates
    intersecting_cells['grid_letter'] = [chr(65 + i) for i in range(len(intersecting_cells))]
    intersecting_cells = intersecting_cells.drop(columns=['index_right'])

    if plot_rst:
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot the original raster
        islands_gdf.plot(ax=ax, color='blue', legend=True, label='Islands')
        # Plot the intersecting grid cells
        intersecting_cells.plot(ax=ax, edgecolor='red', facecolor = 'none', label='Grid cells')
        plt.title("Grid Cells Over Islands")
        plt.legend()
        plt.show()

    return intersecting_cells








import numpy as np
from pathlib import Path

#import CLIMADA modules:
from climada.hazard import Centroids, TropCyclone
from climada.hazard.tc_tracks import TCTracks
from climada.entity import LitPop

#define directories
EXPOSURE_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/exposure")
HAZARD_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard")
TC_TRACKS_DIR = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks/tracks_basins_climada")
STORM_dir = Path("C:/Users/kaibe/Documents/ETH_Zurich/Thesis/Data/hazard/tc_tracks/storm_tc_tracks")

#define countries per tropical cyclone basin according to STORM dataset
NA = [28,44,52,84,132,192,212,214,308,624,328,332,388,659,662,670,740,780]
SI = [174,480,690,626]
SP = [184,242,296,520,570,598,882,90,626,798,548]
WP = [296,584,583,520,585]
EP = [296]
NI = [462]

#create dictionaries for countries per STORM basin
basins_countries = {
    'NA': NA,
    'SI': SI,
    'SP': SP,
    'WP': WP,
    'EP': EP,
    'NI': NI
}

#define variables for exposure
fin = 'gdp' #fin mode
year = 2020 #reference year
res = 30 #resolution in arcsec
buffer = 100000 #buffer in meter

#define variables for TC class
r = 10000 #number of simulated years in tc dataset
freq_corr_STORM = 1 / r





def init_TC_exp(country, load_fls=False, plot_exp=True):

    """Define STORM Basin"""
    for basin, countries in basins_countries.items():
        if country in countries:
            applicable_basin = basin
            print('STORM basin of country: ', applicable_basin)
    if 'applicable_basin' not in locals():
        print('Error: Applicable basin not found - Do not proceed.')
        return 0, 0, 0, 0
    else:
        pass
        

    exp_str = f"Exp_{country}_{fin}_{year}_{res}.hdf5"
    if load_fls and Path.is_file(EXPOSURE_DIR.joinpath(exp_str)):
        """Loading Exposure"""
        print("----------------------Loading Exposure----------------------")
        exp = LitPop.from_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    else:
        """Initiating Exposure"""
        print("----------------------Initiating Exposure----------------------")
        exp = LitPop.from_countries(country, fin_mode=fin, reference_year=year, res_arcsec=res)
        exp.write_hdf5(EXPOSURE_DIR.joinpath(exp_str))
    
    if plot_exp:
        exp.plot_raster()

    """initiate TC hazard from tracks and exposure"""
    # initiate new instance of TropCyclone(Hazard) class:
    haz_str = f"TC_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    track_str = f"Track_sub_{applicable_basin}_{country}_{res}_STORM.hdf5"
    if load_fls and Path.is_file(HAZARD_DIR.joinpath(haz_str)):
        print("----------------------Loading Hazard----------------------")
        tc_storms = TropCyclone.from_hdf5(HAZARD_DIR.joinpath(haz_str))
        storm_basin_sub = TCTracks.from_hdf5(HAZARD_DIR.joinpath(track_str))

    else:
        print("----------------------Generate Hazard----------------------")
        """Generating Centroids"""
        lat = exp.gdf['latitude'].values
        lon = exp.gdf['longitude'].values
        centrs = Centroids.from_lat_lon(lat, lon)
        if plot_exp:
            centrs.plot()

        track_dic = init_STORM_tracks(applicable_basin)

        """Filter TC Tracks"""
        #storm_basin_sub = {}
        #storm_basin_sub = track_dic[applicable_basin].tracks_in_exp(exp, buffer)
        utm_crs = "EPSG:3857" 
        exp_crs = exp.gdf.to_crs(utm_crs) 
        exp_buffer = exp_crs.buffer(distance=buffer, resolution=1)
        exp_buffer = exp_buffer.unary_union
        tc_tracks_lines_crs = track_dic[applicable_basin].to_geodataframe().to_crs(utm_crs)
        print('Next Buffer')
        tc_tracks_lines = tc_tracks_lines_crs.buffer(distance=buffer)
        print('Buffer finsihed')
        select_tracks = tc_tracks_lines.intersects(exp_buffer)
        tracks_in_exp = [track for j, track in enumerate(track_dic[applicable_basin].data) if select_tracks[j]]
        storm_basin_sub = TCTracks(tracks_in_exp)
        print('Filter finished')
        storm_basin_sub.write_hdf5(HAZARD_DIR.joinpath(track_str)) 

        print(f"Number of tracks in {applicable_basin} basin:",storm_basin_sub.size) 

        #generate TropCyclone class from previously loaded TC tracks for one storm data set
        tc_storms = TropCyclone.from_tracks(storm_basin_sub, centroids=centrs)
        tc_storms.frequency = np.ones(tc_storms.event_id.size) * freq_corr_STORM
        tc_storms.check()
        tc_storms.write_hdf5(HAZARD_DIR.joinpath(haz_str))    
    
    return exp, applicable_basin, storm_basin_sub, tc_storms


#Load all STORM tracks for the basin of interest.
def init_STORM_tracks(basin, load_fls=False):
    """Import TC Tracks"""
    all_tracks = []
    storms_basin = {}
    print("----------------------Initiating TC Tracks----------------------")
    fname = lambda i: f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_{i}.txt"
    for i in range(10):
        tracks_STORM = TCTracks.from_simulations_storm(STORM_dir.joinpath(fname(i)))
        all_tracks.extend(tracks_STORM.data)
    tracks_STORM.data = all_tracks
            
    storms_basin[basin] = tracks_STORM

    return storms_basin













import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#import climada stuff
from climada.entity.impact_funcs import trop_cyclone
from climada.engine import ImpactCalc


def init_imp(exp, haz, grid, plot_frequ=True):
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
        imp_grid_evt[i] = imp_grid_csr[i].sum(axis=1) #calculate sum of impacts per grid cell
        imp_grid_evt[i] = [matrix.item() for matrix in imp_grid_evt[i]] #single values per event are stored in 1:1 matrix -> only save value

    #transform matrix to data frame
    imp_grid_evt = pd.DataFrame.from_dict(imp_grid_evt)

    return imp, imp_per_exp, agg_exp, imp_grid_evt




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re

from scipy.optimize import minimize

minimum_payout = 0.1

#Define bounds for minimum and maximum wind speeds
initial_guess = [20, 61]  # Example initial guess
bounds = [(20, 40), (51, 250)]  # Bounds for thresholds

def init_alt_payout(min_trig, max_trig, haz_int, nominal):
    payouts = []
    for i in range(len(haz_int)):
        payout = np.clip((haz_int[i] - min_trig) / (max_trig - min_trig), 0, 1) * nominal
        if payout < (minimum_payout * nominal):
            payout = 0
        payouts.append(payout)
    return payouts


def init_alt_objective_function(params, haz_int, damages, nominal):
    min_trig, max_trig = params
    payouts = init_alt_payout(min_trig, max_trig, haz_int, nominal)
    tot_payout = np.sum(payouts)
    tot_damages = np.sum(damages)
    basis_risk = ((tot_damages - tot_payout)**2)**0.5
    return basis_risk

def init_alt_optimization(haz_int, damages, nominal, print_params=True):
    # Define bounds and initial guesses for each grid cell
    grid_cells = range(len(haz_int.columns)-1)  # Assuming 10 grid cells
    grid_specific_results = {}

    for cell in grid_cells:

        # Perform optimization for each grid cell
        result = minimize(init_alt_objective_function, 
                          initial_guess, 
                          args=(haz_int.iloc[:,cell], damages.iloc[:,cell], nominal), 
                          bounds=bounds, 
                          method='L-BFGS-B')

        optimal_min_speed, optimal_max_speed = result.x
        grid_specific_results[cell] = (optimal_min_speed, optimal_max_speed)

    if print_params:
        print(grid_specific_results)

    #Reshape parameters into a more interpretable form if needed
    optimized_xs = np.array([values[0] for values in grid_specific_results.values()])  #minimum threshold of windspeeds
    optimized_ys = np.array([values[1] for values in grid_specific_results.values()])  #maximum threshold of windspeeds

    return result, optimized_xs, optimized_ys

def alt_pay_vs_damage(damages, optimized_xs, optimized_ys, haz_int, nominal, include_plot=False):
    b = len(damages)
    payout_evt_grd = pd.DataFrame({letter: [None] * b for letter in haz_int.columns[:-1]})
    pay_dam_df = pd.DataFrame({'pay': [0.0] * b, 'damage': [0.0] * b, 'year': [0] * b})

    for i in range(len(damages)):
        tot_dam = np.sum(damages.iloc[i, :])
        pay_dam_df.loc[i,"damage"] = tot_dam
        pay_dam_df.loc[i,"year"] = int(haz_int['year'][i])
        for j in range(len(haz_int.columns)-1):
            grid_hazint = haz_int.iloc[:,j] 
            payouts = init_alt_payout(optimized_xs[j], optimized_ys[j], grid_hazint, nominal)
            payout_evt_grd.iloc[:,j] = payouts
        tot_pay = np.sum(payout_evt_grd.iloc[i, :])
        if tot_pay > nominal:
                tot_pay = nominal
        else: 
            pass
        pay_dam_df.loc[i,"pay"] = tot_pay

    if include_plot:

        mask = pay_dam_df['damage'] <= nominal
        damage_flt = pay_dam_df['damage'][mask]
        payout_flt = pay_dam_df['pay'][mask]

        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 6), gridspec_kw={'width_ratios': [10, 5, 5]})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        ax1.scatter(damage_flt, payout_flt, marker='o', color='blue', label='Events')
        ax1.plot([payout_flt.min(), payout_flt.max()], [payout_flt.min(), payout_flt.max()], color='red', linestyle='--', label='Trendline')
        ax1.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 

        # Add labels and title
        ax1.set_title("Damage vs. Payout for each Event")
        ax1.set_xlabel("Damage [USD]")
        ax1.set_ylabel("Payout [USD]")
        ax1.legend(loc='upper left', borderpad=2.0)

        # Create an inset plot (overview of total data)
        inset_ax1 = inset_axes(ax1, width="30%", height="30%", loc='lower right', borderpad=3.0)  # adjust size and position
        inset_ax1.scatter(pay_dam_df['damage'], pay_dam_df['pay'], label='Overview Data', marker='o', color='blue')
        inset_ax1.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        #inset_ax1.set_xscale('log')
        #inset_ax1.set_yscale('log')
        inset_ax1.set_xlabel("Damage [USD]", fontsize=8)
        inset_ax1.set_ylabel("Payout [USD]", fontsize=8)

        ax2.scatter(damage_flt, payout_flt, marker='o', color='blue', label='Events')
        ax2.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        ax2.set_xscale('log')
        # Add labels and title
        ax2.set_title("Damage vs. Payout for each Event - Low Damages")
        ax2.set_xlabel("Damage [USD]")
        ax2.set_ylabel("Payout [USD]")
        ax2.legend()


        # Create an inset plot (overview of total data)
        inset_ax2 = inset_axes(ax2, width="30%", height="30%", loc='upper left', borderpad=4.0)  # adjust size and position
        inset_ax2.scatter(damage_flt, payout_flt, label='Overview Data', marker='o', color='blue')
        inset_ax2.axhline(y = nominal, color = 'r', linestyle = '-', label='Nominal') 
        inset_ax2.set_xlabel("Damage [USD]", fontsize=8)
        inset_ax2.set_ylabel("Payout [USD]", fontsize=8)
        # Show both plots
        plt.tight_layout()
        plt.show()

    else: 
        pass

    return pay_dam_df














def init_sng_cty_bond(country, prot_share, int_ws=True, incl_plots=False):    
    #load tc_tracks, create hazard class and calculate exposure
    exp, applicable_basins, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex.init_TC_exp(country=country, load_fls=True, plot_exp=incl_plots, plot_centrs=incl_plots, plt_grd=incl_plots)
    #calculate impact and aggregate impact per grid
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=incl_plots) 
    imp_per_event_flt = bpd.init_imp_flt(imp_per_event, lower_rp)
    #set up hazard intensity matrix per grid and event
    if int_ws: 
        int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_stroms=tc_storms, stat=100)
    else:
        int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_tracks=storm_basin_sub)
    #set principal
    premium_dic = {}
    for ps_share in prot_share:
        ps_str = str(ps_share)
        premium_dic[ps_str] = {'ibrd': 0, 'artemis': 0, 'regression': 0, 'required': 0, 'exp_loss': 0, 'att_prob': 0}

    premium_simulation_ps = {}
    returns_ps = {}
    pay_dam_df_ps = {}
    es_metrics_ps = {}

    nom_arr = []
    for i in range(len(prot_share)):
        ps_str = str(prot_share[i])
        nominal, tot_exp, nom_rel_exp = snom.init_nominal(impact=imp, exposure=exp, prot_share=prot_share[i])
        nom_arr.append(nominal)
        #optimize minimum and maximum triggering wind speed per grid cell
        result, optimized_step1, optimized_step2, optimized_step3 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt, damages_evt=imp_per_event_flt)
        #create data frame containing payment vs damage per event
        pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_step1, optimized_step2, optimized_step3, int_grid, nominal, include_plot=incl_plots)
        #calculate expected loss and attachment probability
        exp_loss_ann, att_prob, es_metrics = sb.init_exp_loss_att_prob_simulation(pay_dam_df, nominal)
        #calculate premiums using different approaches
        premium_dic[ps_str]['ibrd'] = prib.monoExp(exp_loss_ann*100, a, k, b) * exp_loss_ann
        premium_dic[ps_str]['artemis'] = exp_loss_ann * artemis_multiplier
        premium_dic[ps_str]['regression'] = cp.calc_premium_regression(exp_loss_ann *100)/100

    #print(f'The premium based on past IBRD bonds is {round(premium_ibrd*100, 3)}%')
    #print(f'The premium based on the artemis multiplier is {round(premium_artemis*100, 3)}%')
    #print(f'The premium based on the regression model from Chatoro et al. 2022 is {round(premium_regression*100, 3)}%')

        #simulate cat bond
        premium_simulation, returns = sb.init_bond_simulation(pay_dam_df, premiums, rf_rates, nominal, ann_com) #simulate cat bond using a Monte Carlo simulation
        #determine premium to match required sharp ratio
        requ_premiums = sb.init_requ_premium(requ_sharpe_ratio, premium_simulation, rf_rates) #calculate required premium to match minimum sharpe ratio
        #sb.display_premiums([requ_premiums], [requ_sharpe_ratio], rf_rates, premium_simulation, exp_loss_ann)   #plot premium versus sharpe ratio
        premium_dic[ps_str]['required'] = requ_premiums
        premium_dic[ps_str]['exp_loss'] = exp_loss_ann
        premium_dic[ps_str]['att_prob'] = att_prob
        premium_simulation_ps[ps_str] = premium_simulation
        returns_ps[ps_str] = returns
        pay_dam_df_ps[ps_str] = pay_dam_df
        es_metrics_ps[ps_str] = es_metrics

    return premium_simulation_ps, returns_ps, premium_dic, nom_arr, pay_dam_df_ps, es_metrics_ps, int_grid, imp_per_event_flt














import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.optimize import minimize

minimum_payout = 0.3
#Define bounds for minimum and maximum wind speeds
initial_guess_ws = [30, 60]  # Wind speed initial guess
initial_guess_cp = [980, 915]  # Central pressure initial guess

def init_alt_payout(min_trig, max_trig, haz_int, nominal, int_haz_cp):
    payouts = []
    min_pay_abs = minimum_payout * nominal
    diff_pay = nominal - min_pay_abs
    if int_haz_cp:
        for i in range(len(haz_int)):
            act_int = haz_int.iloc[i, 0]
            if min_trig >= act_int >= max_trig:
                payout = ((haz_int.iloc[i, 0] - min_trig) / (max_trig - min_trig)) * diff_pay + min_pay_abs
            elif act_int == 0 or act_int > min_trig:
                payout = 0
            else:
                payout = nominal
            payouts.append(payout)
    else:
        for i in range(len(haz_int)):
            act_int = haz_int.iloc[i, 0]
            if min_trig <= act_int <= max_trig:
                payout = ((haz_int.iloc[i, 0] - min_trig) / (max_trig - min_trig)) * diff_pay + min_pay_abs
            elif act_int > max_trig:
                payout = nominal
            else:
                payout = 0
            payouts.append(payout)

    return payouts









import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from rasterio.features import shapes, rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, shape, LineString, GeometryCollection
from shapely.ops import unary_union, split
import shapely


resolution = 1000


def create_islands(exp, crs="EPSG:3857"):
    exp_crs = exp.gdf.to_crs(crs)
    minx, miny, maxx, maxy = exp_crs.total_bounds

    # Calculate the number of rows and columns for the raster
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    # Define the transformation matrix
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Create a generator for the geometries and their associated values
    shapes_gen = ((geom, value) for geom, value in zip(exp_crs.geometry, exp_crs['value']))

    raster = rasterize(
        shapes=shapes_gen,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Fill value for areas with no geometry
        dtype='float32' # Data type of raster
    )

    # Write the raster to a GeoTIFF file using rasterio
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=1,  # Number of bands
            dtype='float32',  # Data type for the raster values
            crs=crs,  # Coordinate reference system
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1)  # Write raster data to the first band
            
            # Read back the raster data from the in-memory file
            raster_data = dataset.read(1)
            transform = dataset.transform

    mask = raster_data > 0  # Create a mask for non-zero values
    # Convert raster mask to polygons (islands)
    cap_style='round'
    shapes_gen = list(shapes(raster_data, mask=mask, transform=transform))
    polygons = [shape(geom) for geom, value in shapes_gen if value > 0]
    # Return as GeoDataFrame
    gdf_islands = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    # Step 3: Merge all adjacent polygons into one using unary_union
    merged_polygon = unary_union(gdf_islands.geometry)

    # Step 4: Create a GeoDataFrame with the merged polygon
    gdf_islands = gpd.GeoDataFrame(geometry=[merged_polygon], crs=crs)
    return gdf_islands

def buffer_islands(islands_gdf, buffer_distance_km, crs="EPSG:3857"):
    # Reproject to a projected CRS to work with distances in meters
    islands_projected = islands_gdf.to_crs(crs)
    
    # Create rectangular buffer
    buffers = islands_projected.geometry.buffer(distance=buffer_distance_km * 1000)  # Convert km to meters
    
    # Return the buffered islands
    return gpd.GeoDataFrame(geometry=buffers, crs=crs)

def divide_into_grid(buffered_gdf, grid_cell_size_km, min_overlap_percent, crs="EPSG:3857"):
    grid_cells = []
    for buffered_island in buffered_gdf.geometry:
        # Create bounding box for the buffered area
        minx, miny, maxx, maxy = buffered_island.bounds

        # Generate grid cells within the bounding box
        for x in np.arange(minx, maxx, grid_cell_size_km * 1000):  # Convert km to meters
            for y in np.arange(miny, maxy, grid_cell_size_km * 1000):
                cell = box(x, y, x + grid_cell_size_km * 1000, y + grid_cell_size_km * 1000)
                if cell.intersects(buffered_island):  # Only include grid cells intersecting the buffered area
                    # Calculate intersection area between the grid cell and the buffered island
                    intersection_area = cell.intersection(buffered_island).area
                    # Calculate total area of the grid cell
                    cell_area = cell.area
                    # Calculate overlap percentage
                    overlap_percent = (intersection_area / cell_area) * 100
                    # Keep the grid cell if the overlap percentage is greater than the threshold
                    if overlap_percent >= min_overlap_percent:
                        grid_cells.append(cell)

    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=crs)
    grid_gdf['grid_letter'] = [chr(65 + i) for i in range(len(grid_gdf))]

    
    # Return as GeoDataFrame
    return grid_gdf

def process_islands(exposure, buffer_distance_km, grid_cell_size_km, min_overlap_percent, plt_true=True):
    exposure_crs = exposure.crs
    islands_gdf = create_islands(exposure)
    buffered_islands = buffer_islands(islands_gdf, buffer_distance_km)
    grid_gdf = divide_into_grid(buffered_islands, grid_cell_size_km, min_overlap_percent)

    islands_gdf = islands_gdf.to_crs(exposure_crs)
    buffered_islands = buffered_islands.to_crs(exposure_crs)
    grid_gdf = grid_gdf.to_crs(exposure_crs)


    if plt_true:
        fig, ax = plt.subplots(figsize=(10, 10))
        islands_gdf.plot(ax=ax, color="green", label="Islands")
        buffered_islands.plot(ax=ax, color="blue", alpha=0.3, label="Buffer")
        grid_gdf.plot(ax=ax, facecolor="none", edgecolor="red", label="Grid Cells")
        handles = [
            plt.Line2D([0], [0], color="green", lw=4, label="Islands"),           
            plt.Line2D([0], [0], color="blue", lw=4, alpha=0.3, label="Buffer"),  
            plt.Line2D([0], [0], color="red", lw=2, label="Grid Cells")           
        ]
        ax.legend(handles=handles, loc="upper right")
        plt.show()
    
    return islands_gdf, buffered_islands, grid_gdf


def divide_islands(islands, num_divisions):
    """Divide a single polygon into equal-sized areas along its longest axis."""
    minx, miny, maxx, maxy = islands.bounds
    split_lines = []
    
    # Split the polygon horizontally or vertically
    if (maxx - minx) > (maxy - miny):
        # Vertical split along x-axis
        x_split_points = np.linspace(minx, maxx, num_divisions+1)
        split_lines = [LineString([(x, miny), (x, maxy)]) for x in x_split_points[1:-1]]
    else:
        # Horizontal split along y-axis
        y_split_points = np.linspace(miny, maxy, num_divisions+1)
        split_lines = [LineString([(minx, y), (maxx, y)]) for y in y_split_points[1:-1]]

    # Split the polygon by the lines
    islands_split = islands
    for line in split_lines:
        islands_split = split(islands, line)

    polygons = []
    # Flatten the resulting Geometrycollection into individual Polygons
    if isinstance(islands_split, GeometryCollection):
        for geom in islands_split.geoms:
            polygons.append(geom)
    else:
        polygons.append(islands_split)

    return polygons


def init_equ_pol(exposure, grid_size=600, crs="EPSG:3857"):
    divided_islands = []
    exposure_crs = exposure.crs
    islands_gdf = create_islands(exposure)
    islands_sng = islands_gdf.explode(index_parts=False)
    buffers = islands_sng.geometry.buffer(distance=200 * 1000)  # Convert km to meters
    z = len(islands_sng)
    islands_un = []
    i = 0
    while i < z - 1:
        if islands_sng.geometry.iloc[i].intersects(buffers.geometry.iloc[i + 1]):
            unioned_polygon = islands_sng.geometry.iloc[i].union(islands_sng.geometry.iloc[i + 1])
            islands_un.append(unioned_polygon)
            i += 2  # Skip next polygon since it's unioned
        else:
            islands_un.append(islands_sng.geometry.iloc[i])
            i += 1

    if i == z - 1:
        islands_un.append(islands_sng.geometry.iloc[i])

    islands_gdf = gpd.GeoDataFrame(geometry=islands_un, crs=crs)

    for i, geometry in enumerate(islands_gdf.geometry):
        # Handle single polygon case
        pol_area = islands_gdf.geometry[i].area / 1000**2
        num_grid = int(pol_area // grid_size)
        if num_grid > 0:
            divided_islands.extend(divide_islands(geometry, num_grid))        
        else:
            divided_islands.append(geometry)
    
    islands_split_gdf = gpd.GeoDataFrame(geometry=divided_islands, crs=crs)

    islands_split_gdf = islands_split_gdf.to_crs(exposure_crs)
    return islands_split_gdf


def sng_bond_nom(arr_nominal, int_grid, imp_admin_evt_flt, imp_per_event_flt, rf_rate, target_sharpe):
    optimized_xs_nom = {}
    optimized_ys_nom = {}
    pay_dam_df_nom = {}
    bond_metrics = {}
    returns = {}
    ann_losses = {}
    es_metrics = {}
    premium_dic = {'ibrd': {}, 'regression': {}, 'required': {}, 'exp_loss': {}, 'att_prob': {}}

    l = len(arr_nominal)
    i = 0

    print_progress_bar(0, l)

    for nom in arr_nominal:
        i+= 1
        nom_str = str(round(nom,0))
        result_nom, optimized_xs_nom[nom_str], optimized_ys_nom[nom_str] = apo.init_alt_optimization(int_grid, nom, damages_grid=imp_admin_evt_flt, print_params=False)

        pay_dam_df_nom[nom_str] = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_xs_nom[nom_str], optimized_ys_nom[nom_str], int_grid, nom, imp_admin_evt_flt)

        exp_loss_ann, att_prob, ann_losses[nom_str], es_metrics[nom_str] = sb.init_exp_loss_att_prob_simulation(pay_dam_df_nom[nom_str], nom, print_prob=False)
        params_ibrd = prib.init_prem_ibrd(want_plot=False)
        a, k, b = params_ibrd
        premium_ibrd = prib.monoExp(exp_loss_ann * 100, a, k, b) * exp_loss_ann
        premium_regression = cp.calc_premium_regression(exp_loss_ann * 100) / 100
        requ_prem = sb.init_prem_sharpe_ratio(ann_losses[nom_str], rf_rate, target_sharpe)
        premium_dic['ibrd'][nom_str] = premium_ibrd
        premium_dic['regression'][nom_str] = premium_regression
        premium_dic['required'][nom_str] = requ_prem
        premium_dic['exp_loss'][nom_str] = exp_loss_ann
        premium_dic['att_prob'][nom_str] = att_prob

        bond_metrics[nom_str], returns[nom_str] = sb.init_bond_simulation(pay_dam_df_nom[nom_str], premium_ibrd, rf_rate, nom)

        print_progress_bar(i, l)

    coverage_nom = []
    basis_risk_nom = []

    for nom_it in arr_nominal:
        nom_str = str(round(nom_it,0))
        coverage_nom.append(bond_metrics[nom_str]['Coverage'])
        basis_risk_nom.append((bond_metrics[nom_str]['Basis_risk'])*-1)

    fig, ax1 = plt.subplots(figsize=(6, 4))

    color = 'tab:red'
    ax1.plot(arr_nominal, coverage_nom, color=color)
    ax1.set_xlabel('Principal [USD]')
    ax1.set_ylabel('Coverage []', color=color)
    ax1.set_ylim(0.3,1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Basis Risk [USD]', color=color)  # we already handled the x-label with ax1
    ax2.plot(arr_nominal, basis_risk_nom, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    premium_ibrd_arr = np.array(get_all_values(premium_dic['ibrd']))
    premium_regression_arr = np.array(get_all_values(premium_dic['regression']))
    premium_required_arr = np.array(get_all_values(premium_dic['required']))


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 4))

    #ax1.plot(prot_share_arr, premium_artemis.values(), label='Artemis')
    ax1.plot(arr_nominal, premium_ibrd_arr, label='IBRD')
    ax1.plot(arr_nominal, premium_regression_arr, label='Regression')
    ax1.plot(arr_nominal, premium_required_arr, label='Sharpe Ratio = 0.5')
    ax1.set_xlabel('Principal [USD]')
    ax1.set_ylabel('Premium [share of principal]')
    ax1.legend(loc='upper right')

    ax2.plot(arr_nominal, (premium_ibrd_arr * arr_nominal), label='IBRD')
    ax2.plot(arr_nominal, (premium_regression_arr * arr_nominal), label='Regression')
    ax2.plot(arr_nominal, (premium_required_arr * arr_nominal), label='Sharpe Ratio = 0.5')
    ax2.set_xlabel('Principal [USDP]')
    ax2.set_ylabel('Premium [USD]')
    ax2.legend(loc='upper left')

    plt.show()

    return bond_metrics, returns, premium_dic, es_metrics, pay_dam_df_nom, optimized_xs_nom, optimized_ys_nom
