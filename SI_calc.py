import mne
#from astropy.stats.circstats import vtest, rayleightest, circmean, circstd
from circular_hist import circular_hist
from mne.time_frequency import tfr_morlet
from scipy.signal import detrend
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from os.path import isdir
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

def gauss_convolve(x, width):
    gaussian = np.exp(x)


if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
chan = "central"
osc_types = ["SO", "deltO"]
#osc_types = ["SO"]
sfreq = 100.
phase_freqs = [(0.5, 1.25),(1.25, 4)]
power_freqs = (15, 18)
conds = ["sham", "fix", "eig"]
title_keys = {"sham":"sham", "fix":"fixed frequency stimulation", "eig":"Eigenfrequency stimulation"}
colors = ["red", "blue", "green"]
osc_cuts = [(-1,1),(-.75,.75)]
gen_crop = (-1, 1)
gauss_win = 0.333

epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
epo.resample(sfreq, n_jobs="cuda")

power_freqs = np.arange(power_freqs[0], power_freqs[1])
power_tfr = tfr_morlet(epo, power_freqs, n_cycles=5, average=False,
                       return_itc=False, n_jobs=n_jobs, output="power")
power_tfr.apply_baseline((-2.25,-1.5))
power_array = power_tfr.data[:,].mean(axis=2)
g = gaussian(int(np.round(gauss_win*power_tfr.info["sfreq"])), 
             gauss_win*power_tfr.info["sfreq"])
for epo_idx in range(len(power_array)):
    power_array[epo_idx,0,] = np.convolve(power_array[epo_idx,0,], g, mode="same")
power_epo = mne.EpochsArray(power_array, epo.info, tmin=-2.25)

SIs = np.empty(len(epo))
spind_max = np.empty(len(epo))
for osc, osc_cut, pf in zip(osc_types, osc_cuts, phase_freqs):
    phase_freq = np.linspace(pf[0], pf[1], 5)
    power_phase_tfr = tfr_morlet(power_epo, phase_freq, n_cycles=1, average=False,
                                 return_itc=False, n_jobs=n_jobs, output="phase")
    phase_tfr = tfr_morlet(epo, phase_freq, n_cycles=1, average=False,
                           return_itc=False, n_jobs=n_jobs, output="phase")


    phase_tfr.crop(tmin=gen_crop[0], tmax=gen_crop[1])
    power_phase_tfr.crop(tmin=gen_crop[0], tmax=gen_crop[1])
    epo.crop(tmin=osc_cut[0], tmax=osc_cut[1])
    this_power = power_epo.copy()
    this_power.crop(tmin=osc_cut[0], tmax=osc_cut[1])
    power = this_power.get_data()[:,0]
    power = detrend(power, axis=1)

    phase = phase_tfr.data[:,0].mean(axis=1)
    power_phase = power_phase_tfr.data[:,0].mean(axis=1)

    osc_inds = np.where(epo.metadata["OscType"]==osc)[0]
    cut_inds = epo.time_as_index((osc_cut[0], osc_cut[1]))

    # calculate SI
    SI = np.mean(np.exp(0+1j*(phase[osc_inds, cut_inds[0]:cut_inds[1]] - \
      power_phase[osc_inds, cut_inds[0]:cut_inds[1]])), axis=1)
    SI_ang = np.angle(SI)

    SIs[osc_inds] = SI_ang

    # calculate SO phase of spindle power maximum
    for osc_idx in np.nditer(osc_inds):
        pma = power[osc_idx, cut_inds[0]:cut_inds[1]].argmax()
        spind_max[osc_idx] = phase[osc_idx, cut_inds[0]:cut_inds[1]][pma]

epo.metadata["SI"] = SIs
epo.metadata["Spind_Max"] = spind_max
epo.save("{}grand_{}_finfo_SI-epo.fif".format(proc_dir, chan), overwrite=True)
