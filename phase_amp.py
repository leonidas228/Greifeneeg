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

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
minmax_freqs = {"SO":(0.5, 1.25), "deltO":(0.75, 4.25)}
chan = "frontal"
osc = "SO"
epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
epo = epo["OscType=='{}'".format(osc)]

pick = mne.pick_channels(epo.ch_names, [chan])[0]
data = epo.get_data()[:,pick,] * 1e+6
fs = epo.info["sfreq"]
low_fq_range = list(np.linspace(minmax_freqs[osc][0],minmax_freqs[osc][1],10))
high_fq_range =  list(np.linspace(10,20,25))

erp = EventRelatedPac(f_pha=minmax_freqs[osc], f_amp=np.linspace(10,20,25),
                      dcomplex="wavelet")
phases = erp.filter(fs, data, ftype='phase', n_jobs=n_jobs)
amplitudes = erp.filter(fs, data, ftype='amplitude', n_jobs=n_jobs)
erpac = erp.fit(phases, amplitudes, method="gc", n_perm=200)
p_vals = erp.infer_pvalues()
erpac_n = erpac.copy()
erpac_n[p_vals==0] = np.nan
erpac_s = erpac.copy()
erpac_s[p_vals==1] = np.nan
plt.figure()
erp.pacplot(erpac_n.squeeze(), epo.times, erp.yvec, cmap="gray")
erp.pacplot(erpac_s.squeeze(), epo.times, erp.yvec)
plt.gca().invert_yaxis()

ph_times = [-0.25,1.75]
ph_inds = epo.time_as_index(ph_times)
pp = PreferredPhase(f_pha=minmax_freqs[osc], f_amp=np.linspace(10,20,25), dcomplex="wavelet")
b_amp, pps, polar_vec = pp.fit(phases[...,ph_inds[0]:ph_inds[1]],
                               amplitudes[...,ph_inds[0]:ph_inds[1]])
ampbin = np.squeeze(b_amp).mean(-1).T
plt.figure()
pp.polar(ampbin, polar_vec, pp.yvec)
