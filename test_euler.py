import exposure_euler as ex_eu

print('Recognized')
if __name__ == "__main__":
    # Define parameters
    print('Started')
    country = "308"

    # Call the function
    exp, applicable_basin, grid_gdf, islands_split_gdf, storm_basin_sub, tc_storms = ex_eu.init_TC_exp(country=country)
    print('Job done')