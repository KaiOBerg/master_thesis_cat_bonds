
import sys
from pathlib import Path
import exposure_cc_euler as ex_cc_eu


def process_country(cty, cc_model):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    ex_cc_eu.init_TC_exp(country=cty, cc_model=cc_model, buffer_distance_km=105, res_exp=30, grid_size=6000, buffer_grid_size=1, load_fls=False, plot_exp=False, plot_centrs=False, plt_grd=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python country_basics_cc_euler.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    cc_model = int(sys.argv[2])
    process_country(country, cc_model)
    print(f"Finished processing country: {country} with model {cc_model}")