import pandas as pd
import numpy as np

term = 3
simulated_years = 10000


def init_equ_nom_sim(events_per_year, nominal_dic_cty):
    ann_loss = np.zeros(term)
    cur_nom_cty = nominal_dic_cty.copy()

    for k in range(term):
        if not events_per_year[k].empty:
            events = events_per_year[k]
            payouts = events['pay'].to_numpy()
            cties = events['country_code'].to_numpy()

            sum_payouts = np.zeros(len(events))

            for idx, (payout, cty) in enumerate(zip(payouts, cties)):
                if payout == 0 or cur_nom_cty[cty] == 0:
                    event_payout = 0
                else:
                    event_payout = payout
                    cur_nom_cty[cty] -= event_payout
                    if cur_nom_cty[cty] < 0:
                        event_payout += cur_nom_cty[cty]
                        cur_nom_cty[cty] = 0
                sum_payouts[idx] = event_payout

            ann_loss[k] = np.sum(sum_payouts)
        else:
            ann_loss[k] = 0

    tot_loss = np.sum(ann_loss)
    return tot_loss



def requ_nom(countries, pay_dam_df_dic, nominal_dic_cty):
    total_losses = []

    for i in range(simulated_years-term):
        events_per_year = []
        for j in range(term):
            events_per_cty = [pay_dam_df_dic[int(cty)].loc[pay_dam_df_dic[int(cty)]['year'] == (i + j)].assign(country_code=cty) for cty in countries]

            year_events_df = pd.concat(events_per_cty, ignore_index=True) if events_per_cty else pd.DataFrame()
            events_per_year.append(year_events_df)

        tot_loss = init_equ_nom_sim(events_per_year, nominal_dic_cty)

        total_losses.append(tot_loss)

    requ_nominal = np.max(total_losses)

    return requ_nominal
