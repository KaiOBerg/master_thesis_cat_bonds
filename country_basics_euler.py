import sys
from pathlib import Path
import exposure_euler as ex_eu

# Directories
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")

def process_country(cty):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    ex_eu.init_TC_exp(country=cty, OUTPUT_DIR=OUTPUT_DIR, STORM_DIR=STORM_DIR, grid_size=6000)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python country_basics_euler.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    process_country(country)
    print(f"Finished processing country: {country}")
