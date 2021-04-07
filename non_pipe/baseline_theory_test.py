import numpy as np

epo_n = 100
sig_len = 500
baselen = 100

signal = np.random.normal(0, 1, size=(epo_n, sig_len))

# mean of subtractions
MoS = np.zeros_like(signal)
for epo_idx in range(epo_n):
    MoS[epo_idx,] = signal[epo_idx,] - signal[epo_idx, :baselen].mean()
MoS_mean = MoS.mean(axis=0)

# subtraction of means
signal_mean = signal.mean(axis=0)
baseline_mean = signal_mean[:baselen]
SoM_mean = signal_mean - baseline_mean.mean()
