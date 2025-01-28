'''Script derives the needed principal/nominal to pretect certain % of GDP or certain return period of damage'''
#prot_rep for principal based on return period
#prot_share for principal based on share of GDP
def init_nominal(exposure, impact=None, prot_rp=None, prot_share=None, print_nom=True):
    tot_exp = exposure.gdf['value'].sum()
    if prot_rp is not None:
        nominal = impact.calc_freq_curve(prot_rp).impact
    else:
        nominal = tot_exp * prot_share
    
    nom_rel_exp = nominal/tot_exp

    if print_nom:
        print(f'The principal of the cat bond is: {round(nominal, 3)} [USD]')
        print(f'Principal as share of GDP: {round(nom_rel_exp, 3)}')


    return nominal



