""" #transform payouts to numpy array
payouts = pay_dam_df['pay'].to_numpy()
damages = pay_dam_df['damage'].to_numpy()

#Parameters for the Monte Carlo Simulation
num_simulations = 10000  # Number of Monte Carlo iterations
simulated_losses = []
simulated_damages = []
event_probability = len(payouts) / r

#Monte Carlo Simulation
for _ in range(num_simulations):
    #randomly generate number of events in one year using poisson distribution and calculated yearly event probability
    num_events = np.random.poisson(lam=event_probability)
    #If there are events in the year, sample that many payouts and the associated damages
    if num_events > 0:
        random_indices = np.random.randint(0, len(payouts), size=num_events)
        simulated_payouts = payouts[random_indices]
        selected_damages = damages[random_indices]
        total_damage = np.sum(selected_damages)
        total_loss = np.sum(simulated_payouts)
        if total_loss > nominal:
            total_loss = nominal
        else: 
            pass
    else:
        total_loss = 0
        total_damage = 0  #No events, no loss damage

    simulated_losses.append(total_loss)
    simulated_damages.append(total_damage)

# Convert simulated losses to a DataFrame
simulated_losses = pd.Series(simulated_losses)
simulated_damages = pd.Series(simulated_damages) """


""" #Expected Loss
exp_loss_hist = pay_dam_df['pay'].sum() / r
exp_loss_sim = simulated_losses.mean()

rel_exp_loss_hist = exp_loss_hist / nominal
rel_exp_loss_sim = exp_loss_sim / nominal

print(f"Expected Loss (historic): {exp_loss_hist}")
print(f"Expected Loss (simulation): {exp_loss_sim}")

print(f"Relative Expected Loss (historic): {rel_exp_loss_hist}")
print(f"Relative Expected Loss (simulation): {rel_exp_loss_sim}")

#Attachment Probability
att_prob_hist = (pay_dam_df['pay'] > 0).sum() / r
att_prob_sim = (simulated_losses > 0).sum() / num_simulations

print(f"Attachment Probability (historic): {att_prob_hist}")
print(f"Attachment Probability (simulation): {att_prob_sim}")

#Coverage
cov_hist = pay_dam_df['pay'].sum() / pay_dam_df['damage'].sum()
cov_sim = sum(simulated_losses) / sum(simulated_damages)

print(f"Coverage (historic): {cov_hist}")
print(f"Coverage (simulation): {cov_sim}")

#Basis Risk
ba_ri_hist = (pay_dam_df['damage'].sum() - pay_dam_df['pay'].sum()) / len(payouts)
ba_ri_sim = (sum(simulated_damages) - sum(simulated_losses)) /len(payouts)

print(f"Basis-Risk (historic): {ba_ri_hist}")
print(f"Basis-Risk (simulation): {ba_ri_sim}")

#Value at Risk
VaR_95 = simulated_losses.quantile(0.95)
VaR_99 = simulated_losses.quantile(0.99)

print(f"Value at Risk (95% confidence): {VaR_95}")
print(f"Value at Risk (99% confidence): {VaR_99}")

#Expected shortfall
ES_95 = simulated_losses[simulated_losses > VaR_95].mean()
ES_99 = simulated_losses[simulated_losses > VaR_99].mean()

print(f"Expected Shortfall (95% confidence): {ES_95}")
print(f"Relative Expected Shortfall (95% confidence): {ES_95 / nominal}")
 """