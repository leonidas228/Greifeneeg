import mne
#from astropy.stats.circstats import vtest, rayleightest, circmean, circstd
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
chan = "central"
osc_types = ["SO", "deltO"]
#osc_types = ["SO"]
sfreq = 50.
phase_freqs = [(0.5, 1.25),(1.25, 4)]
power_freqs = (13, 17)
conds = ["sham", "fix", "eig"]
title_keys = {"sham":"sham", "fix":"fixed frequency stimulation", "eig":"Eigenfrequency stimulation"}
colors = ["red", "blue", "green"]
osc_cuts = [(-1,1),(-.5,.5)]

epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
epo.resample(sfreq, n_jobs="cuda")

power_freqs = np.arange(power_freqs[0], power_freqs[1])
power_tfr = tfr_morlet(epo, power_freqs, n_cycles=5, average=False,
                       return_itc=False, n_jobs=n_jobs, output="power")
power_tfr.crop(tmin=-2, tmax=2)
power = power_tfr.data[:,].mean(axis=2)
power_epo = mne.EpochsArray(power, epo.info)

SIs = np.empty(len(epo))
for osc, osc_cut, pf in zip(osc_types, osc_cuts, phase_freqs):
    phase_freq = np.linspace(pf[0], pf[1], 5)
    power_phase_tfr = tfr_morlet(power_epo, phase_freq, n_cycles=1, average=False,
                                 return_itc=False, n_jobs=n_jobs, output="phase")
    power_phase = power_phase_tfr.data[:,0].mean(axis=1)
    phase_tfr = tfr_morlet(epo, phase_freq, n_cycles=1, average=False,
                           return_itc=False, n_jobs=n_jobs, output="phase")
    phase_tfr.crop(tmin=-2, tmax=2)
    phase = phase_tfr.data[:,0].mean(axis=1)

    osc_inds = np.where(epo.metadata["OscType"]==osc)[0]
    cut_inds = epo.time_as_index((osc_cut[0], osc_cut[1]))

    # calculate SI
    SI = np.mean(np.exp(0+1j*(phase[osc_inds, cut_inds[0]:cut_inds[1]] - \
      power_phase[osc_inds, cut_inds[0]:cut_inds[1]])), axis=1)
    SI_ang = np.angle(SI)

    SIs[osc_inds] = SI_ang

epo.metadata["SI"] = SIs
epo.save("{}grand_{}_finfo_SI-epo.fif".format(proc_dir, chan))
