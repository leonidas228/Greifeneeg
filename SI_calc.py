import mne
#from astropy.stats.circstats import vtest, rayleightest, circmean, circstd
from circular_hist import circular_hist
from mne.time_frequency import tfr_morlet, tfr_multitaper
from scipy.signal import detrend, hilbert
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
phase_freqs = [(0.12, 1.25),(1.25, 4)]
power_freqs = (15, 18)
conds = ["sham", "fix", "eig"]
title_keys = {"sham":"sham", "fix":"fixed frequency stimulation", "eig":"Eigenfrequency stimulation"}
colors = ["red", "blue", "green"]
osc_cuts = [(-1.5,1.5),(-.75,.75)]
gen_crop = (-1, 1)
gauss_win = 0.333
method = "wavelet"

epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
epo.resample(sfreq, n_jobs="cuda")

power_freqs = np.arange(power_freqs[0], power_freqs[1])
if method == "hilbert":
    # epo_power = epo.copy().filter(l_freq=power_freqs[0], h_freq=power_freqs[1])
    # epo_power_data = epo_power.get_data()
    # power = np.zeros_like(epo_power_data)
    # for epo_idx in range(len(epo_power_data)):
    #     power[epo_idx,0,] = np.abs(hilbert(epo_power_data[epo_idx,0,]))
    power_tfr = tfr_multitaper(epo, power_freqs, n_cycles=5, average=False,
                               return_itc=False, n_jobs=n_jobs)
    power_tfr.crop(tmin=-2.15, tmax=1.5) # get rid of edge effects
    power_tfr.apply_baseline((-2.15,-1.5), mode="zscore")
    power = power_tfr.data.mean(axis=2)
else:
    power_tfr = tfr_morlet(epo, power_freqs, n_cycles=5, average=False,
                           return_itc=False, n_jobs=n_jobs, output="power")
    power_tfr.crop(tmin=-2.15, tmax=1.5) # get rid of edge effects
    power_tfr.apply_baseline((-2.15,-1.5), mode="zscore")
    power = power_tfr.data.mean(axis=2)

# g = gaussian(int(np.round(gauss_win*epo.info["sfreq"])),
#              gauss_win*epo.info["sfreq"])
# for epo_idx in range(len(power)):
#     power[epo_idx,0,] = np.convolve(power[epo_idx,0,], g, mode="same")
power_epo = mne.EpochsArray(power, epo.info, tmin=-2.25)

#power = detrend(power, axis=2)
power = power[:,0,]

SIs = np.empty(len(epo))
spind_max = np.empty(len(epo))
for osc, osc_cut, pf in zip(osc_types, osc_cuts, phase_freqs):
    if method == "hilbert":
        epo_phase = epo.copy().filter(l_freq=pf[0], h_freq=pf[1], method="iir",
                                      n_jobs=n_jobs)
        epo_phase_data = epo_phase.get_data()
        phase = np.zeros((epo_phase_data.shape[0],epo_phase_data.shape[-1]))
        epo_power_phase = power_epo.copy().filter(l_freq=pf[0], h_freq=pf[1],
                                                  method="iir", n_jobs=n_jobs)
        epo_power_phase_data = epo_power_phase.get_data()
        power_phase = np.zeros((epo_power_phase_data.shape[0],epo_power_phase_data.shape[-1]))
        for epo_idx in range(len(epo)):
            phase[epo_idx,] = np.angle(hilbert(epo_phase_data[epo_idx,0,]))
            power_phase[epo_idx,] = np.angle(hilbert(epo_power_phase_data[epo_idx,0,]))

    else:
        phase_freq = np.linspace(pf[0], pf[1], 5)
        power_phase_tfr = tfr_morlet(power_epo, phase_freq, n_cycles=1, average=False,
                                     return_itc=False, n_jobs=n_jobs, output="phase")
        phase_tfr = tfr_morlet(epo, phase_freq, n_cycles=1, average=False,
                               return_itc=False, n_jobs=n_jobs, output="phase")

        #phase_tfr.crop(tmin=gen_crop[0], tmax=gen_crop[1])
        #power_phase_tfr.crop(tmin=gen_crop[0], tmax=gen_crop[1])
        phase = phase_tfr.data[:,0].mean(axis=1)
        power_phase = power_phase_tfr.data[:,0].mean(axis=1)

    osc_inds = np.where(epo.metadata["OscType"]==osc)[0]
    crop_epo = epo.copy().crop(tmin=-2.15, tmax=1.5)
    cut_inds = crop_epo.time_as_index((osc_cut[0], osc_cut[1]))

    # calculate SI
    SI = np.mean(np.exp(0+1j*(phase[osc_inds, cut_inds[0]:cut_inds[1]] - \
      power_phase[osc_inds, cut_inds[0]:cut_inds[1]])), axis=1)
    SI_ang = np.angle(SI)

    SIs[osc_inds] = SI_ang

    # calculate SO phase of spindle power maximum
    for osc_idx in np.nditer(osc_inds):
        pma = power[osc_idx, cut_inds[0]:cut_inds[1]].argmax()
        spind_max[osc_idx] = phase[osc_idx, cut_inds[0]:cut_inds[1]][pma]

    if osc == "SO":
        fig, axes = plt.subplots(1,3,figsize=(38.4,21.6))
        # epo_phase = epo.copy().filter(l_freq=pf[0], h_freq=pf[1], method="iir",
        #                               n_jobs=n_jobs)
        epo_data = epo.get_data()[:,0,]
        axes[0].plot(epo.times[cut_inds[0]:cut_inds[1]],
                     epo_data[osc_inds,cut_inds[0]:cut_inds[1]].T,
                     alpha=0.01, color="blue")
        axes[0].set_title("SO")
        axes[1].plot(epo.times[cut_inds[0]:cut_inds[1]],
                     phase[osc_inds,cut_inds[0]:cut_inds[1]].T,alpha=0.01,
                     color="red")
        axes[1].set_title("SO Instantaneous Phase")
        axes[2].plot(epo.times[cut_inds[0]:cut_inds[1]],
                     power[osc_inds,cut_inds[0]:cut_inds[1]].T,alpha=0.1,
                     color="green")
        axes[2].set_title("SO Spindle Power")
        fig.suptitle("{} transform".format(method))


epo.metadata["SI"] = SIs
epo.metadata["Spind_Max"] = spind_max
epo.save("{}grand_{}_finfo_SI-epo.fif".format(proc_dir, chan), overwrite=True)
