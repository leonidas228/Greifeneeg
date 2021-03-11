import mne
from tensorpac import EventRelatedPac as ERPAC
import pandas as pd
import numpy as np
from os.path import isdir
import matplotlib.pyplot as plt
import pickle
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)


if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 4
chan = "central"
osc_types = ["SO", "deltO"]
#osc_types = ["SO"]
sfreq = 100.
phase_freqs = [(0.16, 1.25),(1.25, 4)]
power_freqs = (5, 20)
conds = ["sham", "eig", "fix"]
durs = ["30s", "2m", "5m"]
osc_cuts = [(-1.25,1.25),(-.75,.75)]
method = "wavelet"

epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
epo.resample(sfreq, n_jobs="cuda")

osc_types = ["SO", "deltO"]
epos = []
dfs = []
for osc, osc_cut, pf in zip(osc_types, osc_cuts, phase_freqs):
    f_amp = np.linspace(power_freqs[0],power_freqs[1],30)
    ep = ERPAC(f_pha=[pf[0], pf[1]], f_amp=f_amp, dcomplex=method)

    this_epo = epo["OscType == '{}'".format(osc)]
    this_df = this_epo.metadata.copy()
    cut_inds = this_epo.time_as_index((osc_cut[0], osc_cut[1]))
    data = this_epo.get_data()[:,0,] * 1e+6

    phase = ep.filter(this_epo.info["sfreq"], data, ftype="phase", n_jobs=n_jobs)
    power = ep.filter(this_epo.info["sfreq"], data, ftype="amplitude", n_jobs=n_jobs)

    power = power[...,cut_inds[0]:cut_inds[1]]
    phase = phase[...,cut_inds[0]:cut_inds[1]]
    times = this_epo.times[cut_inds[0]:cut_inds[1]]

    erpac = ep.fit(phase, power, mcp="fdr")
    plt.figure(figsize=(38.4,21.6))
    ep.pacplot(erpac.squeeze(), times, ep.yvec, pvalues=ep.pvalues)
    plt.title("{} ERPAC, phase at {}-{}Hz ({} transform)".format(osc, pf[0],
                                                                 pf[1], method))
    plt.ylabel("Hz", fontdict={"size":24})
    plt.xlabel("Time (s)", fontdict={"size":24})

    evo = this_epo.average().data[0,cut_inds[0]:cut_inds[1]]
    evo = (evo - evo.min())/(evo.max()-evo.min())
    evo = evo*5 + 10

    plt.plot(times, evo, linewidth=10, color="gray", alpha=0.8)

    plt.savefig("../images/ERPAC_{}_{}.png".format(osc, method))
