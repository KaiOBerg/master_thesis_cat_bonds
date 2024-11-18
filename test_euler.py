import exposure_euler as ex_eu

print('Recognized')
if __name__ == "__main__":
    # Define parameters
    print('Started')
    countries = [242, 388, 192, 626]

    for cty in countries:
        exp, applicable_basin, grid_gdf, islands_split_gdf, storm_basin_sub, tc_storms = ex_eu.init_TC_exp(country=cty)
    print('Job done')