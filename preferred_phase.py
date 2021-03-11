import mne
from tensorpac import PreferredPhase as PP
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
phase_freqs = [(0.16, 1.25),(1.25, 4)]
power_freqs = (10, 20)
conds = ["sham", "eig", "fix"]
durs = ["30s", "2m", "5m"]
osc_cuts = [(-1.25,1.25),(-.75,.75)]
method = "hilbert"
bins = 48

epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
epo.resample(sfreq, n_jobs="cuda")

osc_types = ["SO", "deltO"]

epos = []
for osc, osc_cut, pf in zip(osc_types, osc_cuts, phase_freqs):
    plt.figure(figsize=(38.4,21.6))
    pp = PP(f_pha=(pf[0], pf[1]),
            f_amp=np.linspace(power_freqs[0],power_freqs[1],20),
            dcomplex=method)

    this_epo = epo["OscType == '{}'".format(osc)]
    df = this_epo.metadata
    cut_inds = this_epo.time_as_index((osc_cut[0], osc_cut[1]))
    data = this_epo.get_data()[:,0,] * 1e+6
    data = data[...,cut_inds[0]:cut_inds[1]]

    phase = pp.filter(this_epo.info["sfreq"], data, ftype="phase", n_jobs=n_jobs)
    phase = phase.mean(axis=0, keepdims=True)
    power = pp.filter(this_epo.info["sfreq"], data, ftype="amplitude", n_jobs=n_jobs)

    subplot_muster = 330
    for dur in durs:
        for cond in conds:
            inds = np.where((df["StimType"]==cond) & (df["Dur"]==dur))[0]
            ampbin, pps, vecbin = pp.fit(phase[:,inds], power[:,inds], n_bins=bins)
            ampbin = ampbin.mean(-1).squeeze()
            subplot_muster += 1
            #interp = .2 if osc=="SO" else .01
            interp = .2
            pp.polar(ampbin.T, vecbin, pp.yvec, cmap='hot', interp=interp,
                     subplot=subplot_muster, cblabel='Amplitude bins')
            plt.gca().collections[-1].colorbar.remove()
            plt.title("{} {}\n".format(cond, dur))
    plt.suptitle("{} ({} transform)".format(osc, method))
    #plt.tight_layout()
    plt.savefig("../images/pp_{}_{}".format(osc, method))
