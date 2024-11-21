import exposure_euler as ex_eu
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

print('Recognized')
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")

def process_country(cty):
    """Wrapper function to process a single country."""
    return ex_eu.init_TC_exp(country=cty, OUTPUT_DIR=OUTPUT_DIR, STORM_DIR=STORM_DIR)

if __name__ == "__main__":
    # Define parameters
    print('Started')
    countries = [882]  # List of country codes

    # Use ProcessPoolExecutor to parallelize across countries
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_country, countries))

    print('Job done')