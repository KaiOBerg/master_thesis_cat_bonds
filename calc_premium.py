'''This script is used to calculate premiums based on the paper of Chatoro et al., 2022 '''

#define TLM equation

Term = 36 #months
IG = 0 #most bonds are not -> high yield bonds not investment graded
Hybrid = 0 #refers to trigger
GCIndex = 180 #Guy Carpenter Global Property Catastrophe Rate on Line Index (January, 2025)
BBSpread = 1.6 #ICE BofA BB US High Yield Index Option-Adjusted Spread (January, 2025)

#values extracted from respective publication
b_0 = -0.5907
b_1 = 1.3986
b_2 = 2.2520
b_3 = 0.0377
b_4 = 0.4613
b_5 = -0.0239
b_6 = -2.6742
b_7 = 0.7057

#return premiums
def calc_premium_regression(expected_loss, Peak_Multi=0):
    P = b_0 + b_1 * expected_loss + b_2 * Peak_Multi + b_3 * GCIndex + b_4 * BBSpread + b_5 * Term + b_6 * IG + b_7 * Hybrid
    return P

