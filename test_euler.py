import exposure_euler as ex_eu

if __name__ == "__main__":
    # Define parameters
    country = "308"

    # Call the function
    exp, applicable_basin, grid_gdf, islands_split_gdf, storm_basin_sub, tc_storms = ex_eu.init_TC_exp(country)