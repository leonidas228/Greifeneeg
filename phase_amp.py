import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
from scipy.signal import hilbert
from mne.time_frequency import psd_multitaper, tfr_morlet
import matplotlib.pyplot as plt
import pandas as pd
from tensorpac import Pac, EventRelatedPac, PreferredPhase
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 24}
matplotlib.rc('font', **font)


if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 2
minmax_freqs = {"SO":(0.6, 1.25), "deltO":(0.75, 4.25)}
power_freqs = np.linspace(8,24,25-8)
chan = "central"
osc = "SO"
phase_freqs = np.linspace(minmax_freqs[osc][0], minmax_freqs[osc][1], 2)
epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
epo = epo["OscType=='{}'".format(osc)]

#epo = epo["Cond=='{}' or Cond=='{}'".format("eig5m", "fix5m")]
#epo = epo["Cond=='sham'"]

pick = mne.pick_channels(epo.ch_names, [chan])[0]
data = epo.get_data()[:,pick,] * 1e+6
fs = epo.info["sfreq"]
low_fq_range = list(np.linspace(minmax_freqs[osc][0],minmax_freqs[osc][1],10))
high_fq_range =  list(np.linspace(10,20,25))

erp = EventRelatedPac(f_pha=phase_freqs, f_amp=power_freqs,
                      dcomplex="wavelet")

phases = erp.filter(fs, data, ftype='phase', n_jobs=n_jobs)
amplitudes = erp.filter(fs, data, ftype='amplitude', n_jobs=n_jobs)
time_inds = epo.time_as_index((-1,1))
phases = phases[...,time_inds[0]:time_inds[1]]
amplitudes = amplitudes[...,time_inds[0]:time_inds[1]]

erpac = erp.fit(phases, amplitudes, method="gc", n_perm=200)
p_vals = erp.infer_pvalues()
for p_freq in range(erpac.shape[1]):
    erpac_n = erpac[:,p_freq,].copy()
    erpac_n[p_vals[:,p_freq,]==0] = np.nan
    erpac_s = erpac[:,p_freq,].copy()
    erpac_s[p_vals[:,p_freq,]==1] = np.nan
    plt.figure()
    erp.pacplot(erpac_n.squeeze(), epo.times[time_inds[0]:time_inds[1]],
                erp.yvec, cmap="gray")
    erp.pacplot(erpac_s.squeeze(), epo.times[time_inds[0]:time_inds[1]],
                erp.yvec)
    plt.gca().invert_yaxis()
    plt.ylabel("Hz", fontsize=24)
    plt.xlabel("Time (s)", fontsize=24)
    plt.title("EPRAC 0.6-1.2Hz", fontsize=24)

# tensorpac
# p = Pac(f_pha=low_fq_range, f_amp=high_fq_range, dcomplex="wavelet")
# phases = p.filter(fs, data, ftype='phase', n_jobs=n_jobs)
# amplitudes = p.filter(fs, data, ftype='amplitude', n_jobs=n_jobs)
# p.idpac = (6, 3, 4)
# pac = p.fit(phases, amplitudes, n_perm=200)
# pac_avg = pac.mean(-1)
# pvalues = p.infer_pvalues(p=0.05, mcp="maxstat")
# pac_avg_ns = pac_avg.copy()
# pac_avg_ns[pvalues<0.05] = np.nan
# p.comodulogram(pac_avg_ns, cmap='gray')
# pac_avg_s = pac_avg.copy()
# pac_avg_s[pvalues>0.05] = np.nan
# p.comodulogram(pac_avg_s, cmap='Spectral_r')

pp = PreferredPhase(f_pha=phase_freqs, f_amp=power_freqs, dcomplex="wavelet")
b_amp, pps, polar_vec = pp.fit(phases, amplitudes)
for p_freq in range(b_amp.shape[2]):
    ampbin = b_amp[:,:,p_freq].mean(-1).T
    plt.figure()
    pol_ax = pp.polar(ampbin, polar_vec, power_freqs)

for p_freq in range(len(phases)):
    plt.figure()
    plt.plot(epo.times[time_inds[0]:time_inds[1]],phases[p_freq,].T,
             color="blue", alpha=0.01)
