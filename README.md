Description taken from `description_code_sam.docx`

The heart of the code is saved in 
    • Simulate_bond.py
    • Simulate_multi_cty_bond.py
The first one is for single country bonds and the second one is for multi country bonds. The function “init_exp_loss_att_prob_simulation” is used to simulate important metrics of the bond such as expected loss and attachment probability so that the premiums can be calculated based on its results. This function uses “init_bond_exp_loss” for each term of the bond. 
The function “init_bond_simulation” simulates the bond with the premiums and uses “init_bond” for each term of the bond. 
In the same python script, there is also the function “find_sharpe” which estimates the premium necessary to meet a certain sharpe ratio based on the simulations of “init_exp_loss_att_prob_simulation”.

Another important python script is
    • alt_pay_opt.py
The first functions are used to optimize the payouts. The problem we discussed about last time, that the optimization is not stable for high principals, is solved. The problem was that the minimum payout of the bond was in percentage. So, if the minimum payout of a bond with a high principal was higher than the minimum damage, I want to protect the coverage decreased. I now changed the minimum payout to an absolute value which works fine. The functions are a bit messy because I tried to write them in a way that the payout can also be calibrated with central pressures instead of wind speed, though I did not test yet if it works appropriately because I always use wind speeds.
The last function in the script calculates then the total payout per event which is also very important for the bond simulation.

Another important python script is
    • exposure.py
This script is used to calculate exposure, import tracks, filter tracks, generate TC class and create the TC track boundary as well as subareas. So, in a sense this function creates the base of simulation for each country. This script also uses a function from grider.py. In this script are a few functions currently not used because I am still deciding what’s the best way to create subareas for each country.

There are a few other important scripts such as 
    • haz_int_grd.py -> extracts the hazard intensity per grid
    • impact.py -> calculates impact per event and subarea and so on

There are a lot of other scripts on github, but not all of them are currently used. The ones I named in this document are the core of my product and if they work the results should be fine. The other notebooks are mostly only simple functions without much complicated code. 
The jupyter notebooks are in constant evolution and only use functions from the scripts. Its difficult to check them because I always try out new stuff so often they do not run properly. If you want to have a look at one, it makes sense to use those from the climada workshop because they work probably.
