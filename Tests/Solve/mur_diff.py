import numpy as np


L = 12753.290567641216
k = -8.532096937497737
B0 = 1.3703368533854523


def sigmoid(B):
    exp_term = np.exp(-k * (B - B0))
    sig = L / (1 + exp_term)
    d_sig = (L * k * exp_term) / ((1 + exp_term)**2)
    return sig, d_sig



B_sat = 1.9
dmu_dB = sigmoid(B_sat)[1]
mu_r = sigmoid(B_sat)[0]
mu_rd = mu_r/(1 - B_sat/mu_r*dmu_dB)
print(mu_rd)




