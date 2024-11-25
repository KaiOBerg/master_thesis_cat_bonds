
import sys
from pathlib import Path
import exposure_cc_euler as ex_cc_eu


def process_country(cty):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    ex_cc_eu.init_TC_exp(country=cty, cc_model='CMCC', buffer_distance_km=105, res_exp=30, grid_size=6000, buffer_grid_size=1, load_fls=False, plot_exp=False, plot_centrs=False, plt_grd=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python country_basics_euler.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    process_country(country)
    print(f"Finished processing country: {country}")