import exposure_buffer_test as ex_buf
import impact as cimp
import bound_prot_dam as bpd
import alt_pay_opt as apo
import set_nominal as snom
import haz_int_grd as hig
import exposures_alt as aexp
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

OUTPUT_DIR = Path("/cluster/work/climate/kbergmueller/cty_data")
STORM_DIR = Path("/cluster/work/climate/kbergmueller/storm_tc_tracks")
lower_share = 0.045
country = 882
buffer_distance_arr = np.arange(5, 151, 5)  # Buffer distances
prot_rp = 250
int_stat = np.arange(10, 101, 10)


def sng_cty_bond(country, grid_specs, int_stat, prot_share, file_path, storm_path, buffer_distance, to_prot_share=None, incl_plots=False):    
    #load tc_tracks, create hazard class and calculate exposured
    exp, applicable_basin, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = aexp.init_TC_exp(country=country, grid_specs=grid_specs, file_path=file_path, storm_path=storm_path,
                                                                                              buffer_distance_km=buffer_distance, load_fls=True, plot_exp=incl_plots, plot_centrs=incl_plots, plt_grd=incl_plots)
    imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf, plot_frequ=incl_plots) 
    imp_per_event_flt, imp_admin_evt_flt, imp_lower_rp = bpd.init_imp_flt(imp_per_event, imp_admin_evt, prot_share=to_prot_share, exposure=exp)
    
    nominal = snom.init_nominal(impact=imp, exposure=exp, prot_share=prot_share, print_nom=False)

    basis_risk_dic = {}
    for i in int_stat:
        int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_storms=tc_storms, stat=i)
        result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt_flt, damages_evt=imp_per_event_flt, print_params=incl_plots)
        pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_1, optimized_2, int_grid, nominal, imp_admin_evt)
        basis_risk_dic[i] = np.sum(pay_dam_df['damage']) - np.sum(pay_dam_df['pay'])

    int_grid = hig.init_haz_int(grid_gdf, admin_gdf, tc_storms=tc_storms, stat='mean')
    result, optimized_1, optimized_2 = apo.init_alt_optimization(int_grid, nominal, damages_grid=imp_admin_evt_flt, damages_evt=imp_per_event_flt, print_params=incl_plots)
    pay_dam_df = apo.alt_pay_vs_damage(imp_per_event_flt, optimized_1, optimized_2, int_grid, nominal, imp_admin_evt)
    basis_risk_dic['mean'] = np.sum(pay_dam_df['damage']) - np.sum(pay_dam_df['pay'])

    basis_risk_df = pd.DataFrame([basis_risk_dic])

    return basis_risk_df, len(optimized_1), exp


def process_buffer(buffer):
    """Process a single buffer distance."""
    try:
        exp, applicable_basins, grid_gdf, admin_gdf, storm_basin_sub, tc_storms = ex_buf.init_TC_exp(
            country=country, buffer_distance_km=buffer, file_path=STORM_DIR
        )
        
        imp, imp_per_event, imp_admin_evt = cimp.init_imp(exp, tc_storms, admin_gdf)
        nominal = snom.init_nominal(impact=imp, exposure=exp, prot_share=lower_share)
        count = imp_per_event[imp_per_event >= nominal].shape[0]
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

    max_count = df[df['Number Events'] == df['Number Events'].max()].iloc[0]
    # Extract the buffer value
    selected_buffer = max_count['buffer']

    all_dfs = []

    for i in range(10):

        grid_specs = {
            0: [1+i,1+i], 
            1: [1+i,1+i] 
        }

        br, y, exp = sng_cty_bond(country, grid_specs, int_stat, prot_rp=prot_rp, to_prot_share=lower_share, buffer_distance=selected_buffer, file_path=OUTPUT_DIR, storm_path=STORM_DIR, incl_plots=False)
        br['Count grids'] = y
        all_dfs.append(br)

    combined_br = pd.concat(all_dfs, ignore_index=True)

    combined_br.to_excel(OUTPUT_DIR / f"basis_risk_grids_{country}.xlsx", index=False)


    print("Job done")
