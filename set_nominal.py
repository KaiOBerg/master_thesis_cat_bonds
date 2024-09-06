from climada.engine import ImpactCalc

upper_rp = 200
rel_prot = 0.05

def init_nominal(impact=None, exposure=None):
    tot_exp = exposure.gdf['value'].sum()
    if impact is not None:
        nominal = impact.calc_freq_curve(200).impact
    else:
        nominal = tot_exp * rel_prot
    
    nom_rel_exp = nominal/tot_exp


    print(f'The principal of the cat bond is: {nominal}')
    print(f'Principal as perecntage of GDP: {nom_rel_exp}')


    return nominal, tot_exp, nom_rel_exp



