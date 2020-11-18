import mne
from mne.time_frequency import tfr_morlet, read_tfrs
from os import listdir
from os.path import isdir
import re
import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy.stats import circmean, circvar, circstd
import matplotlib.pyplot as plt
import pandas as pd
from tensorpac import EventRelatedPac
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
minmax_freqs = {"SO":(0.6, 1.25), "deltO":(0.75, 4.25)}
chan = "central"
osc = "SO"
power_freqs = np.linspace(8,24,25-8)
phase_freqs = np.linspace(minmax_freqs[osc][0], minmax_freqs[osc][1], 3)

epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
epo = epo["OscType=='{}'".format(osc)]

data = epo.get_data(picks=chan)[:,0] * 1e+6
fs = epo.info["sfreq"]

# # use erpac for phase/power
# erp = EventRelatedPac(f_pha=phase_freqs, f_amp=power_freqs,
#                       dcomplex="wavelet")
# phases = erp.filter(fs, data, ftype='phase', n_jobs=n_jobs)
# amplitudes = erp.filter(fs, data, ftype='amplitude', n_jobs=n_jobs)

tfr = read_tfrs("{}grand_central-tfr.h5".format(proc_dir))[0]
amplitudes = tfr.data[:,0,] * 1e+10
tfr_ph = tfr_morlet(epo, phase_freqs, n_cycles=1, average=False,
                 return_itc=False, n_jobs=n_jobs, output="complex")
phases = np.angle(tfr_ph.data[:,0,])
epo.crop(tmin=tfr.times[0], tmax=tfr.times[-1])

twins = {"Second":{"twin":(0.1, 0.4),"freq":(13,16)}}

df = epo.metadata.copy()
for k,v in twins.items():
    time_range = v["twin"]
    freq_range = v["freq"]
    time_inds = epo.time_as_index(time_range)
    freq_inds = np.where((power_freqs >= freq_range[0]) & (power_freqs <= freq_range[1]))[0]
    ph = phases[...,time_inds[0]:time_inds[1]].mean(axis=1)
    amp = amplitudes[:, freq_inds, time_inds[0]:time_inds[1]].mean(axis=1)
    avg_amps = []
    avg_phs = []
    for epo_idx in range(len(epo)):
        avg_phs.append(ph[epo_idx, :].mean())
        avg_amps.append(amp[epo_idx, :].mean())
    df["{}Phase".format(k)] = avg_phs
    df["{}Amp".format(k)] = avg_amps

amp_diffs = []
for epo_idx in range(len(epo)):
    (idx0, idx1) = epo.time_as_index((0.15,0.75))
    idx_cent = epo.time_as_index(0)
    peak_time_inds = find_peaks(data[epo_idx,idx0:idx1])[0] + idx0
    peak_amp = np.max(data[epo_idx,peak_time_inds])
    amp_diffs.append(peak_amp - data[epo_idx,idx_cent][0])
df["AmpDiff"] = amp_diffs

df.to_pickle("{}phase_amp_{}".format(proc_dir, osc))

df_circ_dict = {"Subj":[], "Cond":[], "Count":[]}
for k in twins.keys():
    df_circ_dict["{}PhaseMean".format(k)] = []
    df_circ_dict["{}PhaseStd".format(k)] = []
subjs = list(np.unique(df["Subj"].values))
conds = list(np.unique(df["Cond"].values))
for subj in subjs:
    for cond in conds:
        temp_df = df.query("Subj=='{}' and Cond=='{}'".format(subj, cond))
        for k in twins.keys():
            c_mean = circmean(temp_df["{}Phase".format(k)].values, high=np.pi, low=-np.pi)
            c_std = circstd(temp_df["{}Phase".format(k)].values, high=np.pi, low=-np.pi)
            df_circ_dict["{}PhaseMean".format(k)].append(c_mean)
            df_circ_dict["{}PhaseStd".format(k)].append(c_std)
        df_circ_dict["Count"].append(len(temp_df))
        df_circ_dict["Subj"].append(subj)
        df_circ_dict["Cond"].append(cond)

df_circ = pd.DataFrame.from_dict(df_circ_dict)
df_circ.to_pickle("{}phase_variance_{}".format(proc_dir, osc))
