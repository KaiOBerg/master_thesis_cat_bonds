'''Script to calculate parametric index for CAT bonds per subarea'''
import pandas as pd
import numpy as np
import geopandas as gpd
import re
from shapely.geometry import Point
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt



def init_haz_int(grid=None, admin=None, tc_storms=None, tc_tracks=None, stat=100, cc_model=None):
    """
    Calculates a specified statistic (mean, percentiles) for each events sustained wind speeds
    from tc_storms/tc_tracks for each grid cell. Use tc_track if parametric index is central pressure and 
    tc_strom if parametric index is wind speed.

    Args:
        tc_storms: A hazard object containing wind speed data per centroid.
        tc_tracks: A hazard object containing central pressure data per along tc track.
        grid: Input subareas as grid cells
        admin: Input subareas as administrative boundaries
        stat (str or float): The statistic to calculate. Can either be a numer to calculate percentile or the
                    string 'mean' to calculate the average.
        cc_model: input global circulation model if you want to derive statistics for climate change scenarios
    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics with labels as columns.
    """

    if tc_storms:
        hazard = tc_storms.centroids.gdf
        hazard = hazard.to_crs(admin.crs)
        centrs_to_grid = hazard.sjoin(admin, how='left', predicate="within")
        agg_exp = centrs_to_grid.groupby('admin_letter').apply(lambda x: x.index.tolist())

        int_grid = {letter: [None] * len(tc_storms.event_id) for letter in agg_exp.keys()}
        int_grid['year'] = [None] * len(tc_storms.event_id)
        int_grid['month'] = [None] * len(tc_storms.event_id)


        #Iterate over each event
        for i in range(len(tc_storms.event_id)):
            year_string = tc_storms.event_name[i]
            month_string = tc_storms.get_event_date()[i]
            int_grid['year'][i] = extract_year(year_string, cc_model)
            int_grid['month'][i] = int(month_string.split('-')[1])
            #For each grid cell, calculate the desired statistic
            for letter, line_numbers in agg_exp.items():
                selected_values = tc_storms.intensity[i, line_numbers]

                if stat == 'mean':
                    int_grid[letter][i] = selected_values.mean()
                elif isinstance(stat, (int, float)):
                    dense_array = selected_values.toarray()
                    flattened_array = dense_array.flatten()
                    int_grid[letter][i] = np.percentile(flattened_array, stat)
                else:
                    raise ValueError("Invalid statistic choice. Choose number for percentile or 'mean'")        
        int_grid = pd.DataFrame.from_dict(int_grid)
        int_grid['count_grids'] = (int_grid > 0).sum(axis=1)
        int_grid.loc[int_grid['count_grids'] > 0, 'count_grids'] -= 2

    elif tc_tracks:
        grid_crs = grid.crs

        int_grid = {letter: [None] * len(tc_tracks.data) for letter in grid['grid_letter']}
        int_grid['year'] = [None] * len(tc_tracks.data)
        int_grid['month'] = [None] * len(tc_tracks.data)
        for i in range(len(tc_tracks.data)):
            latitudes = tc_tracks.data[i]['lat'].values
            longitudes = tc_tracks.data[i]['lon'].values
            central_pressures = tc_tracks.data[i]['central_pressure'].values
            geometry = [Point(lon, lat) for lon, lat in zip(longitudes, latitudes)]
            points_gdf = gpd.GeoDataFrame(pd.DataFrame({'Central Pressure (mb)': central_pressures}), geometry=geometry).set_crs(grid_crs)
            points_in_grid = gpd.sjoin(points_gdf, grid)

            year_string = tc_tracks.data[i].attrs['sid']
            month_string = tc_tracks.data[i].coords['time'][0]
            int_grid['year'][i] = extract_year(year_string, cc_model)
            int_grid['month'][i] = month_string.dt.month.item()
            max_pressure_per_grid = points_in_grid.groupby('grid_letter')['Central Pressure (mb)'].min()
            for letter in grid['grid_letter']:
                if max_pressure_per_grid.get(letter):
                    int_grid[letter][i] = max_pressure_per_grid.get(letter)
                else:
                    int_grid[letter][i] = 0
        int_grid = pd.DataFrame.from_dict(int_grid)
        int_grid['count_grids'] = (int_grid > 0).sum(axis=1)
        int_grid.loc[int_grid['count_grids'] > 0, 'count_grids'] -= 1



    else: 
        print("Error: Input missing")
        
    return int_grid


#determine year for every event and add it to dataframe
#Regular expression to capture the number before .txt and the one after .txt-
pattern = r'(\d+)\.txt-(\d+)'
pattern_cc = r'(\d+)_IBTRACSDELTA\.txt-(\d+)'

'''Extract the year of event'''
def extract_year(string, cc_model):
    if cc_model is not None:
        match = re.search(pattern_cc, string)
    else:
        match = re.search(pattern, string)
    if match:
        before_txt = int(match.group(1))
        after_txt = int(match.group(2))
        year = before_txt * 1000 + after_txt
    return year

'''plot the intensity and damage for each event'''
def plt_int_dam(imp_admin_evt, int_grid):

    num_grids = len(imp_admin_evt.columns)
    n_cols = 2  
    n_rows = math.ceil(num_grids / n_cols) 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 8 * n_rows))

    if n_rows > 1:
        axes = axes.flatten()

    for i, (columnName, columnData) in enumerate(imp_admin_evt.items()):
        corr, pval = spearmanr(columnData, int_grid[columnName])
        ax = axes[i]

        ax.scatter(columnData, int_grid[columnName], marker='o', edgecolor='blue', facecolor='none', label='Events')
        ax.set_title(f"Damage vs. Wind Speed - Island {columnName}", fontsize=18)
        ax.set_xlabel("Damage [USD]", fontsize=16)
        ax.set_ylabel("Wind Speed [m/s]", fontsize=16)
        ax.text(0, 68, f'Spearman.: {round(corr, 2)}', fontsize = 16)
        ax.set_ylim(0,80)
        ax.legend(loc='lower right')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()