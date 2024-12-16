import sys
from pathlib import Path
import exposure_euler as ex_eu
import impact as cimp
import set_nominal as snom

# Directories
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")

def process_country(cty):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    exp, applicable_basin, grid_gdf, storm_basin_sub, tc_storms = ex_eu.init_TC_exp(country=cty, OUTPUT_DIR=OUTPUT_DIR, STORM_DIR=STORM_DIR, crs="EPSG:3832")
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, plot_frequ=False) 
    nominal = snom.init_nominal(impact=imp, exposure=exp, prot_rp=250)
    return nominal

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python country_basics_euler.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    print(f"Processing country: {country}")
    nominal = process_country(country)
    print(country, nominal)
    print(f"Finished processing country: {country}")
