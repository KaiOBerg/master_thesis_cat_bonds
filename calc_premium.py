'''This script is used to calculate premiums based on the paper of Chatoro et al., 2022 '''

#define TLM equation

Peal_Multi = 0
Term = 36
IG = 1
Hybrid = 0
GCIndex = 200.8
BBSpread = 1.83
EL = 0.05

b_0 = -0.5907
b_1 = 1.3986
b_2 = 2.2520
b_3 = 0.0377
b_4 = 0.4613
b_5 = -0.0239
b_6 = -2.6742
b_7 = 0.7057

P = b_0 + b_1 * EL + b_2 * Peal_Multi + b_3 * GCIndex + b_4 * BBSpread + b_5 * Term + b_6 * IG + b_7 * Hybrid

print('The expected premium is:', P)

print(P/EL)