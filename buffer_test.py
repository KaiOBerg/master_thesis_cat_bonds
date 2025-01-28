'''used to test which TC boundary distance is the most appropriate for Samoa'''
#intended for use on Euler cluster

import exposure_buffer_test as ex_buf
import impact as cimp
import set_nominal as snom
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
#cat bond set ups
lower_share = 0.05
country = 882
#Tc boundary distance
buffer_distance_arr = np.arange(10, 151, 10)  # Buffer distances



def process_buffer(buffer):
    """Process a single buffer distance."""
    try:
        exp, applicable_basins, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex_buf.init_TC_exp(
            country=country, buffer_distance_km=buffer, file_path=STORM_DIR
        )
        
        imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=False)
        nominal = snom.init_nominal(impact=imp, exposure=exp, prot_share=lower_share)
        count = imp_per_event[imp_per_event >= nominal].shape[0]
        print(f'Process finished for buffer {buffer}')
        return buffer, count
    except Exception as e:
        print(f"Error processing buffer {buffer}: {e}")
        return buffer, None

if __name__ == "__main__":
    print("Started")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_buffer, buffer_distance_arr))

    valid_results = [(buf, cou) for buf, cou in results if cou is not None]
    df = pd.DataFrame(valid_results, columns=["Buffer Distance (km)", "Number Events"])
    output_file = OUTPUT_DIR / "events_vs_buffer_150.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Saved results to {output_file}")

