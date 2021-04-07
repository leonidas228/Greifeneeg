import mne
from mne.stats.cluster_level import _find_clusters
from tensorpac import EventRelatedPac as ERPAC
from tensorpac import PreferredPhase as PP
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
osc_types = ["SO"]
sfreq = 200.
phase_freqs = [(0.16, 1.25),(1.25, 4)]
power_freqs = (10, 20)
conds = ["sham", "eig", "fix"]
conds = ["fix"]
durs = ["30s", "2m", "5m"]
osc_cuts = [(-1.5,1.5),(-.75,.75)]
method = "wavelet"
exclude = ["002", "003", "028"]

f_amp = np.linspace(power_freqs[0],power_freqs[1],50)
epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
for excl in exclude:
    epo = epo["Subj!='{}'".format(excl)]
epo.resample(sfreq, n_jobs="cuda")

epos = []
dfs = []
for osc, osc_cut, pf in zip(osc_types, osc_cuts, phase_freqs):
    osc_epo = epo["OscType == '{}'".format(osc)]
    for cond in conds:
        cond_epo = osc_epo["StimType == '{}'".format(cond)]
        ep = ERPAC(f_pha=[pf[0], pf[1]], f_amp=f_amp, dcomplex=method)

        this_df = cond_epo.metadata.copy()
        cut_inds = cond_epo.time_as_index((osc_cut[0], osc_cut[1]))
        data = cond_epo.get_data()[:,0,] * 1e+6

        phase = ep.filter(cond_epo.info["sfreq"], data, ftype="phase", n_jobs=n_jobs)
        power = ep.filter(cond_epo.info["sfreq"], data, ftype="amplitude", n_jobs=n_jobs)

        power = power[...,cut_inds[0]:cut_inds[1]]
        phase = phase[...,cut_inds[0]:cut_inds[1]]
        times = cond_epo.times[cut_inds[0]:cut_inds[1]]

        erpac = ep.fit(phase, power, mcp="fdr")
        plt.figure(figsize=(38.4,21.6))
        ep.pacplot(erpac.squeeze(), times, ep.yvec, pvalues=ep.pvalues)
        plt.title("{} ERPAC {}, phase at {}-{}Hz ({} transform)".format(osc,
                                                                        cond,
                                                                        pf[0],
                                                                        pf[1],
                                                                        method))
        plt.ylabel("Hz", fontdict={"size":24})
        plt.xlabel("Time (s)", fontdict={"size":24})

        evo = cond_epo.average().data[0,cut_inds[0]:cut_inds[1]]
        evo = (evo - evo.min())/(evo.max()-evo.min())
        evo = evo*3 + 13

        plt.plot(times, evo, linewidth=10, color="gray", alpha=0.8)

        plt.savefig("../images/ERPAC_{}_{}_{}.png".format(osc, cond, method))

        ## get clusters of significant points and examine them
        # make a time index map of the erpac
        time_idx_map = np.tile(np.arange(erpac.shape[-1]), (erpac.shape[0], 1))

        p_thr = (ep.pvalues.squeeze()<0.05).astype(int)
        clusters, sums = _find_clusters(p_thr, 0.99)
        for clust in clusters:
            clu = np.reshape(clust, time_idx_map.shape)
            inds = np.sort(np.unique(time_idx_map[clu]))
            p = PP(f_pha=[pf[0], pf[1]], f_amp=f_amp)
            ampbin, pp, vecbin = p.fit(phase[...,inds], power[...,inds],
                                        n_bins=72)
            pp = np.squeeze(pp).T
            ampbin = np.squeeze(ampbin).mean(-1)
            plt.figure(figsize=(38.4,21.6))
            p.polar(ampbin.T, vecbin, p.yvec, cmap="hot", vmin=None, vmax=None,
                    interp=.1)
            plt.suptitle("Preferred Phase at {:.2f} - {:.2f}s".format(times[inds[0]],
                                                                      times[inds[-1]]))
            plt.tight_layout()
