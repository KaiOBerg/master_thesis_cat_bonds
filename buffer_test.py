import exposure_buffer_test as ex_buf
import impact as cimp
import set_nominal as snom
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
lower_share = 0.05
country = 882
buffer_distance_arr = np.arange(0, 151, 15)  # Buffer distances



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

    # Parallelize processing of buffer distances
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_buffer, buffer_distance_arr))

    # Filter out failed results and prepare DataFrame
    valid_results = [(buf, cou) for buf, cou in results if cou is not None]
    df = pd.DataFrame(valid_results, columns=["Buffer Distance (km)", "Number Events"])
    # Save the DataFrame to an Excel file
    output_file = OUTPUT_DIR / "events_vs_buffer.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Saved results to {output_file}")

