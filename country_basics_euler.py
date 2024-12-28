import sys
from pathlib import Path
import exposure_euler as ex

STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks/climate_change")
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller")


def process_country(cty, cc_model):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    ex.init_TC_exp(country=cty, file_path=OUTPUT_DIR, storm_path=STORM_DIR)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python country_basics_cc_euler.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    cc_model = sys.argv[2]
    process_country(country, cc_model)
    print(f"Finished processing country: {country} with model {cc_model}")