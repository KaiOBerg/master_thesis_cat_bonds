from climada.engine import ImpactCalc

def init_nominal(impact, exposure, prot_rp=None, prot_share=None, print_nom=True):
    tot_exp = exposure.gdf['value'].sum()
    if prot_rp is not None:
        nominal = impact.calc_freq_curve(prot_rp).impact
    else:
        nominal = tot_exp * prot_share
    
    nom_rel_exp = nominal/tot_exp

    if print_nom:
        print(f'The principal of the cat bond is: {nominal}')
        print(f'Principal as percentage of GDP: {nom_rel_exp}')


    return nominal, tot_exp, nom_rel_exp


