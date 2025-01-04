import sys
from pathlib import Path
import exposures_cc as ex_cc

STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks/climate_change")
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data_cc")
countries_150 = [332, 388, 214, 44, 548, 192, 84, 90, 598, 626] 
fiji = [242]
countries_30 = [480, 212, 670, 28, 52, 662, 659, 308, 882, 780, 570, 776, 174, 184, 584, 585, 798, 132, 583]


def process_country(cty, cc_model, res, crs):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    ex_cc.init_TC_exp(country=cty, cc_model=cc_model, file_path=OUTPUT_DIR, storm_path=STORM_DIR, buffer_distance_km=105, res_exp=res, grid_size=6000, buffer_grid_size=1, crs=crs, load_fls=False, plot_exp=False, plot_centrs=False, plt_grd=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python country_basics_cc_euler.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    cc_model = sys.argv[2]
    if country in countries_150:
        resolution = 150
        crs='EPSG:3857'
    elif country in countries_30:
        resolution = 30
        crs='EPSG:3857'
    elif country in fiji:
        resolution = 150
        crs = 'EPSG:3832'
    process_country(country, cc_model, resolution, crs)
    print(f"Finished processing country: {country} with model {cc_model}")