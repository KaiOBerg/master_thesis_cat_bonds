import exposure_buffer_test as ex_buf
import impact as cimp
import set_nominal as snom
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
lower_share = 0.045
country = 882
buffer_distance_arr = np.arange(5, 151, 5)  # Buffer distances
file_loc = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
print("Recognized")

def process_buffer(buffer):
    """Process a single buffer distance."""
    try:
        exp, applicable_basins, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex_buf.init_TC_exp(
            country=country, buffer_distance_km=buffer, file_path=file_loc
        )
        print(applicable_basins)
        imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf)
        nominal = snom.init_nominal(impact=imp, exposure=exp, prot_share=lower_share)
        return buffer, nominal
    except Exception as e:
        print(f"Error processing buffer {buffer}: {e}")
        return buffer, None

if __name__ == "__main__":
    print("Started")

    # Parallelize processing of buffer distances
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_buffer, buffer_distance_arr))

    # Filter out failed results and prepare DataFrame
    valid_results = [(buf, nom) for buf, nom in results if nom is not None]
    df = pd.DataFrame(valid_results, columns=["Buffer Distance (km)", "Nominal"])
    # Save the DataFrame to an Excel file
    output_file = OUTPUT_DIR / "nominal_vs_buffer.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Saved results to {output_file}")

    print("Job done")
