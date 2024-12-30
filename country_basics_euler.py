import sys
from pathlib import Path
import exposure_euler as ex

STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
countries_150 = [332, 388, 214, 44, 548, 192, 84, 90, 598, 626] 
fiji = [242]
countries_30 = [480, 212, 670, 28, 52, 662, 659, 308, 882, 780, 570, 776, 174, 184, 584, 585, 798, 132, 583]


def process_country(cty, res, crs):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    ex.init_TC_exp(country=cty, res=res, OUTPUT_DIR=OUTPUT_DIR, STORM_DIR=STORM_DIR, crs=crs)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python country_basics_cc_euler.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    if country in countries_150:
        resolution = 150
        crs='EPSG:3857'
    elif country in countries_30:
        resolution = 30
        crs='EPSG:3857'
    elif country in fiji:
        resolution = 150
        crs = 'EPSG:3832'
    process_country(country, resolution, crs)
    print(f"Finished processing country: {country}; Resolution: {resolution}; CRS: {crs}")