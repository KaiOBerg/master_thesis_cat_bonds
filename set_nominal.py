from climada.engine import ImpactCalc

def init_nominal(impact, exposure, prot_rp=None, prot_share=None):
    tot_exp = exposure.gdf['value'].sum()
    if impact is not None:
        nominal = impact.calc_freq_curve(prot_rp).impact
    else:
        nominal = tot_exp * prot_share
    
    nom_rel_exp = nominal/tot_exp


    print(f'The principal of the cat bond is: {nominal}')
    print(f'Principal as perecntage of GDP: {nom_rel_exp}')


    return nominal, tot_exp, nom_rel_exp



