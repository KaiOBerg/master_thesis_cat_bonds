## Corresponding Code for the Master-Thesis: Enhancing Countries’ Access to Natural Disaster Insurance by Addressing Current Challenges in the Catastrophe Bond Market by Kai Bergmüller (2025)

The construction of the CAT bond proposed for Samoa in the first case study of the thesis is saved in:
   * single_country_bond_main.ipynb

   The notebook depends on various python scripts each dedicated to a specific part of the CAT bond design. The scripts are the following:\
      * exposures_alt.py -> derive exposure, TC boundary, subareas, and hazard data; this script makes use of\
         * grider.py -> created TC boundary to filter for eligible hazards\
      * impact.py -> takes the hazard and subarea data and derives the damage per event and subarea\
      * bound_prot_dam.py -> calculates return periods and sets damages below minimum payout to zero\
      * haz_int_grd.py -> takes the hazard and subarea data and derives the parametric index per event and subarea (wind speed or central pressure)\
      * set_nominal.py -> takes the hazard and damage data and derives the nominal/principal based on return periods of damage or the share of GDP\
      * alt_pay_opt.py -> takes the damage,hazard, and parametric index data and calibrates the payout function and results in a dataframe indicating damage, payout, year, and month per hazard event\
      * simulate_bond.py -> used to simulate the bond using all hazard events. First, the losses as well as the relevant metrics for the losses will be derive (expected loss, attachment probability, and various financial metrics)\
      * calc_prem.py; simulate_bond.py; prem_ibrd.py -> used to derive premiums using various pricing methods based on loss data. calc_prem.py = Chatoro-Pricing; simulate_bond.py = Benchmark-Pricing; prem_ibrd.py = IBRD-Pricing\
      * simulate_bond.py -> using the premium estimates now the returns of the bond can be simulated and the bond set up is complete

   Testing of the bond was done by using:\
      * sng_cty_cc.py -> for testing climate change resilience\
      which made use of the following script to get exposure and hazard data:\
         * exposures_cc.py 
   and\
      * product_test_historic.ipynb -> tests product with historic and perturbed tracks
   
   Addtionally, to derive optimal product combinations (subareas and parametric index statistic) the following script was used:\
      * opt_sub_area_euler.py\
   and to test TC boundary distance:\
      * buffer_test.py\
      which made use of a reduced version of exposures_alt.py saved in:\
         * exposure_buffer_test.py

   The dependance of the optimization of the payout function on the number of events was tested in:\
      * testing_optimization.ipynb\
   and the test regarding the sequence of years in the bond set up was done in:\
      * test_simulation_years.ipynb




Results for the pooling set up described in Case Study 2 are presented in:
   * results_main_pool_sub.ipynb\
   The notebook first imports the data on the respective single country bonds which were calculated on euler with the script
      * master_of_disaster_pooling.py
      This script makes use of the script:
         * n_fct_t_rl_thm_ll.py\
      which includes all the wrapper function to design single-country or multi-country bonds making use of
         * simulate_multi_cty_bond.py -> simulate multi-country bonds used for derive losses and related metrics as well as returns\
      premiums were calculated with the same scripts as in Case Study 1\
      Tranches were calculated with the script:
         * functions.py\
   Optimal pools for the set of countries were cacluated with:
      * pooling_n_pools.py\
      based on the optimization function from Ciullo et al. (2022) and Elsener (2024) saved in:
         * pooling_functions_ciullo.py

          


Results for the insurance scheme described in Case Study 3 are presented in:
   * results_fs_pools.ipynb\
   The notebook first imports the data on the respective single country bonds which were calculated on euler with the script
      * master_of_disaster_pooling.py\
      This script makes use of the script:
         * n_fct_t_rl_thm_ll.py\
      which includes all the wrapper function to design single-country or multi-country bonds making use of
         * simulate_multi_cty_bond.py -> simulate multi-country bonds used for derive losses and related metrics as well as returns\
      premiums were calculated with the same scripts as in Case Study 1\
      Tranches were calculated with the script:
         * functions.py\
   Optimal pools for the set of countries were cacluated with:
      * pooling_max_nom.py\
      based on the optimization function from Ciullo et al. (2022) and Elsener (2024) saved in:
         * pooling_functions_ciullo.py\
   The share of premium donation calculations is presented in:
      * plot_fs_share_pay.ipynb\
      and a automated function to derive premium shares was implemeted in:
         * functions.py\
   The historical risk-return performance of the Norwegian Pension fund was derived in:
      * risk_return_profile.ipynb 




TC data was created with the following functions on euler:
   * IBTRACS_import.py
   * country_basics_euler.py
   using
      * exposure_euler.py
   * country_basics_cc_euler.py\
   using
      * exposures_cc_euler.py



Other scripts:\
For each country damage for the 250-year return period event was derived with:
   * assess_tc_danger.py\
And to test the bond simulating functions the following script was used:
   * hands_on_example_simulate_bond.py\
An analysis what happens to IBRD-Pricing premiums when pooling is presented in:
   * prem_ibrd_analysis.ipynb



Other notebooks which were not used for final results but could be useful in the future:
   * wang_transformation.ipynb -> very popular method to calculate reinsurance prices
   * model_sofr.ipynb -> implementing Cox-Ingersoll-Ross Model to simulate risk free rate




For the practical part during the CLIMADA days I used the following notebooks:
   * climada_days_notebook.ipynb
   * backup_climada_days_notebook.ipynb
   using functions from the follwowing script:
      * climada_days_functions.py
      * exposure_climada.py
In case there is the necessity for another similar presentation those notebook could be used as basics




In Zenodo under the DOI: 10.5281/zenodo.14879604 (https://doi.org/10.5281/zenodo.14879604) generated data during Case Study 1 is saved.
Additionally, the files necessary to run Case Study 2 and 3 are included to allow to run the code without generating all the hazard and exposure data first. 
Moreover, a power-point presentation is saved, which accompanies the script hands_on_example_simulate_bond.py.

