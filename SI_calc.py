import mne
from circular_hist import circular_hist
from mne.time_frequency import tfr_morlet
from scipy.signal import resample
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from os.path import isdir
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
chans = ["central"]
osc_types = ["SO", "deltO"]
#osc_types = ["SO"]
sfreq = 50.
epo_pref = "ak_"
epo_pref = ""
phase_freqs = [(0.5, 1.25),(1.25, 4)]
power_freqs = (12, 15)
conds = ["sham", "fix30s", "eig30s"]
colors = ["red", "blue", "green"]
osc_cuts = [(-1,1),(-.5,.5)]


power_freqs = np.arange(power_freqs[0], power_freqs[1])

fig, axes = plt.subplots(len(osc_types), len(conds), subplot_kw={"projection":"polar"})
for osc_idx, osc in enumerate(osc_types):
    phase_freq = np.linspace(phase_freqs[osc_idx][0], phase_freqs[osc_idx][1], 5)
    epochs = mne.read_epochs("{}{}grand_{}-epo.fif".format(proc_dir, epo_pref, chans[0]),
                          preload=True)
    epochs = epochs["OscType=='{}'".format(osc)]
    epochs.resample(sfreq, n_jobs="cuda")
    for cond_idx, cond in enumerate(conds):
        epo = epochs["Cond=='{}'".format(cond)]
        phase_tfr = tfr_morlet(epo, phase_freq, n_cycles=1, average=False,
                           return_itc=False, n_jobs=n_jobs, output="phase")
        phase_tfr.crop(tmin=-2, tmax=1.5)
        phase = phase_tfr.data[:,0].mean(axis=1)

        power_tfr = tfr_morlet(epo, power_freqs, n_cycles=5, average=False,
                               return_itc=False, n_jobs=n_jobs, output="power")
        power_tfr.crop(tmin=-2, tmax=1.5)
        power = power_tfr.data[:,].mean(axis=2)
        power_epo = mne.EpochsArray(power, epo.info)
        power_phase_tfr = tfr_morlet(power_epo, phase_freq, n_cycles=1, average=False,
                                     return_itc=False, n_jobs=n_jobs, output="phase")
        power_phase = power_phase_tfr.data[:,0].mean(axis=1)

        epo.crop(tmin=-2, tmax=1.5)
        cut_inds = epo.time_as_index((osc_cuts[osc_idx][0],osc_cuts[osc_idx][1]))

        # calculate SI
        SI = np.mean(np.exp(0+1j*(phase - power_phase)), axis=1)
        SI_ang = np.angle(SI)

        # plot
        if len(osc_types) > 1:
            circular_hist(axes[osc_idx][cond_idx], SI_ang, color=colors[cond_idx],
                          alpha=0.6)
            axes[osc_idx][cond_idx].set_title("{} {}".format(osc, cond))
        else:
            circular_hist(axes[cond_idx], SI_ang, color=colors[cond_idx],
                          alpha=0.6)
            axes[cond_idx].set_title("{} {}".format(osc, cond))
