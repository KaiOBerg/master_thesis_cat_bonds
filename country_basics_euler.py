import sys
from pathlib import Path
import n_fct_t_rl_thm_ll as bnd_fct

# Directories
OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
IBRD_DIR = Path("/cluster/work/climate/kbergmueller")

def process_country(cty):
    """Wrapper function to process a single country."""
    print(f"Processing country: {cty}")
    bnd_all = bnd_fct.sng_cty_bond(cty, res_exp=150, grid_size=10000, grid_specs=[4,4], buffer_grid_size=5, prot_rp=250, to_prot_share=0.045, storm_dir=STORM_DIR, output_dir=OUTPUT_DIR, ibrd_path=IBRD_DIR, incl_plots=True, plt_save=True)
    return bnd_all

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python country_basics_euler.py <country_code>")
        sys.exit(1)
    
    country = int(sys.argv[1])
    print(f"Processing country: {country}")
    bnd_all = process_country(country)
    print(country, bnd_all)
    print(f"Finished processing country: {country}")
