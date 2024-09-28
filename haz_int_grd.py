import pandas as pd
import numpy as np
import geopandas as gpd
import re
from shapely.geometry import Point



def init_haz_int(grid=None, admin=None, tc_storms=None, tc_tracks=None, stat=100):
    """
    Calculates a specified statistic (mean, max, or median) for each events sustained wind speeds
    from tc_storms for each grid cell.

    Args:
        tc_storms: A hazard object containing wind speed data per centroid.
        agg_exp (dict): A dictionary where keys are the labels of grid cells and values are lists of 
                        line numbers corresponding to indices in tc_storms.intensity.
        stat (str): The statistic to calculate. Can either be a numer to calculate percentile or the
                    string 'mean' to calculate the average.
    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics with labels as columns.
    """

    if tc_storms:
        #group each exposure point according to grid cell letter
        centrs_to_grid = tc_storms.centroids.gdf.sjoin(admin, how='left', predicate="within")
        agg_exp = centrs_to_grid.groupby('admin_letter').apply(lambda x: x.index.tolist())

        #Initialize a dictionary to hold the calculated statistics
        int_grid = {letter: [None] * len(tc_storms.event_id) for letter in agg_exp.keys()}
        int_grid['year'] = [None] * len(tc_storms.event_id)

        #Iterate over each event
        for i in range(len(tc_storms.event_id)):
            year_string = tc_storms.event_name[i]
            int_grid['year'][i] = extract_year(year_string)
            #For each grid cell, calculate the desired statistic
            for letter, line_numbers in agg_exp.items():
                selected_values = tc_storms.intensity[i, line_numbers]

                # Calculate the statistic based on the user's choice
                if stat == 'mean':
                    int_grid[letter][i] = selected_values.mean()
                elif isinstance(stat, (int, float)):
                    dense_array = selected_values.toarray()
                    flattened_array = dense_array.flatten()
                    int_grid[letter][i] = np.percentile(flattened_array, stat)
                else:
                    raise ValueError("Invalid statistic choice. Choose number for percentile or 'mean'")        
        int_grid = pd.DataFrame.from_dict(int_grid)
        int_grid = int_grid.where(int_grid >= 33, 0)
        int_grid['count_grids'] = (int_grid > 0).sum(axis=1)
        int_grid.loc[int_grid['count_grids'] > 0, 'count_grids'] -= 1

    elif tc_tracks:
        grid_crs = grid.crs

        #Initialize a dictionary to hold the calculated statistics
        int_grid = {letter: [None] * len(tc_tracks.data) for letter in grid['grid_letter']}
        int_grid['year'] = [None] * len(tc_tracks.data)
        for i in range(len(tc_tracks.data)):
            latitudes = tc_tracks.data[i]['lat'].values
            longitudes = tc_tracks.data[i]['lon'].values
            central_pressures = tc_tracks.data[i]['central_pressure'].values
            geometry = [Point(lon, lat) for lon, lat in zip(longitudes, latitudes)]
            points_gdf = gpd.GeoDataFrame(pd.DataFrame({'Central Pressure (mb)': central_pressures}), geometry=geometry).set_crs(grid_crs)
            points_in_grid = gpd.sjoin(points_gdf, grid)

            year_string = tc_tracks.data[i].attrs['sid']
            int_grid['year'][i] = extract_year(year_string)
            max_pressure_per_grid = points_in_grid.groupby('grid_letter')['Central Pressure (mb)'].min()
            for letter in grid['grid_letter']:
                if max_pressure_per_grid.get(letter):
                    int_grid[letter][i] = max_pressure_per_grid.get(letter)
                else:
                    int_grid[letter][i] = 0
        int_grid = pd.DataFrame.from_dict(int_grid)
        #int_grid = int_grid.where(int_grid < 979, 0)
        int_grid['count_grids'] = (int_grid > 0).sum(axis=1)
        int_grid.loc[int_grid['count_grids'] > 0, 'count_grids'] -= 1



    else: 
        print("Error: Input missing")
        
    return int_grid


#determine year for every event and add it to dataframe
#Regular expression to capture the number before .txt and the one after .txt-
pattern = r'(\d+)\.txt-(\d+)'

# Function to extract the numbers
def extract_year(string):
    match = re.search(pattern, string)
    if match:
        before_txt = int(match.group(1))
        after_txt = int(match.group(2))
        year = before_txt * 1000 + after_txt
    return year
