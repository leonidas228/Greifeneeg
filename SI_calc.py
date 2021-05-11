import mne
from tensorpac import Pac
from scipy.signal import detrend
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
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
sfreq = 100.
phase_freqs = [(0.5, 1.25),(1.25, 4)]
power_freqs = [(12, 15), (15, 18)]
conds = ["sham", "fix", "eig"]
title_keys = {"sham":"sham", "fix":"fixed frequency stimulation", "eig":"Eigenfrequency stimulation"}
colors = ["red", "blue", "green"]
osc_cuts = [(-1.25,1.25),(-.75,.75)]
gauss_win = 0.1
method = "wavelet"
baseline = None
bl_time = (-2.35, -1.25)
power_detrend = False
power_win = True
convolve = False

epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
epo.resample(sfreq, n_jobs="cuda")

osc_types = ["SO", "deltO"]

epos = []
for osc, osc_cut, pf in zip(osc_types, osc_cuts, phase_freqs):
    this_epo = epo["OscType == '{}'".format(osc)]
    for power_freq in power_freqs:
        p = Pac(f_pha=(pf[0], pf[1]), #f_pha=np.linspace(pf[0],pf[1],10),
                f_amp=power_freq,
                dcomplex=method)

        cut_inds = this_epo.time_as_index((osc_cut[0], osc_cut[1]))
        bl_inds = this_epo.time_as_index((bl_time[0], bl_time[1]))
        data = this_epo.get_data()[:,0,] * 1e+6

        phase = p.filter(this_epo.info["sfreq"], data, ftype="phase", n_jobs=n_jobs)
        power = p.filter(this_epo.info["sfreq"], data, ftype="amplitude", n_jobs=n_jobs)
        power_phase = p.filter(this_epo.info["sfreq"], power.mean(axis=0),
                               ftype="phase", n_jobs=n_jobs)

        if baseline == "zscore":
            bl_m = power[...,bl_inds[0]:bl_inds[1]].mean(axis=-1, keepdims=True)
            bl_std = power[...,bl_inds[0]:bl_inds[1]].std(axis=-1, keepdims=True)
            power = (power - bl_m) / bl_std

        phase = phase.mean(axis=0)
        power = power.mean(axis=0)
        power_phase = power_phase.mean(axis=0)
        phase = phase[:, cut_inds[0]:cut_inds[1]]
        power = power[:, cut_inds[0]:cut_inds[1]]
        power_phase = power_phase[:, cut_inds[0]:cut_inds[1]]
        data = data[:, cut_inds[0]:cut_inds[1]]
        times = this_epo.times[cut_inds[0]:cut_inds[1]]

        if power_detrend:
            power = detrend(power)

        if power_win:
            power = power * tukey(power.shape[-1], 0.5)

        if convolve:
            g = gaussian(int(np.round(convolve*this_epo.info["sfreq"])),
                         convolve*epo.info["sfreq"])
            for epo_idx in range(len(power)):
                power[epo_idx,] = np.convolve(power[epo_idx,], g, mode="same")

        # calculate SI
        SI = np.mean(np.exp(0+1j*(phase[:, cut_inds[0]:cut_inds[1]] - \
          power_phase[:, cut_inds[0]:cut_inds[1]])), axis=1)
        SI_ang = np.angle(SI)

        # calculate SO phase of spindle power maximum
        maxima = np.argmax(power, axis=1)
        spind_max = phase[[np.arange(len(phase))],[maxima]][0,]

        if osc == "SO":
            fig, axes = plt.subplots(1,4,figsize=(38.4,21.6))
            axes[0].plot(times, data.T, alpha=0.005, color="blue")
            axes[0].plot(times, data.mean(axis=0), alpha=1, color="black")
            axes[0].set_title("SO")
            axes[0].set_ylim((-100, 100))
            axes[1].plot(times, phase.T, alpha=0.005, color="red")
            axes[1].plot(times, phase.mean(axis=0), alpha=1, color="black")
            axes[1].set_title("SO Instantaneous Phase")
            vmax = 5 if method=="hilbert" else 200
            axes[2].imshow(power, aspect="auto", vmin=0, vmax=vmax)
            axes[2].set_title("SO Spindle Power")
            axes[2].set_xticks(np.arange(25,250,50))
            axes[2].set_xticklabels(times[np.arange(25,250,50)])
            axes[3].hist(times[maxima], bins=20)
            axes[3].set_title("SO Spindle Power Maxima")
            fig.suptitle("{} transform".format(method))
            plt.savefig("../images/SO_{}_pha_pow_max.png".format(method))

        this_epo.metadata["SI"] = SI_ang
        this_epo.metadata["Spind_Max_{}-{}Hz".format(power_freq[0],
                                                     power_freq[1])] = spind_max
    epos.append(this_epo)

epo = mne.concatenate_epochs(epos)
epo.save("{}grand_{}_finfo_SI-epo.fif".format(proc_dir, chan), overwrite=True)
