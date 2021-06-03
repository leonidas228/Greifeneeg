import mne
from mne.stats.cluster_level import _find_clusters
from joblib import Parallel, delayed
from mne.stats import fdr_correction
from tensorpac import EventRelatedPac as ERPAC
from mne.time_frequency import tfr_morlet
#from tensorpac import PreferredPhase as PP
import pandas as pd
import numpy as np
from scipy.stats import norm
from os.path import isdir
import matplotlib.pyplot as plt
import pickle
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 40}
matplotlib.rc('font', **font)

def do_erpac(ep, epo, cut, baseline=None, fit_args={"mcp":"fdr", "p":0.05,
                                                    "n_jobs":1,
                                                    "method":"circular",
                                                    "n_perm":1000}):
    data = epo.get_data()[:,0,] * 1e+6
    phase = ep.filter(epo.info["sfreq"], data, ftype="phase", n_jobs=n_jobs)
    power = ep.filter(epo.info["sfreq"], data, ftype="amplitude", n_jobs=n_jobs)

    if baseline:
        base_inds = epo.time_as_index((baseline[0], baseline[1]))
        bl = power[...,base_inds[0]:base_inds[1]]
        bl_mu = bl.mean(axis=-1, keepdims=True)
        bl_std = bl.std(axis=-1, keepdims=True)
        power = (power - bl_mu) / bl_std

    cut_inds = epo.time_as_index((cut[0], cut[1]))

    power = power[...,cut_inds[0]:cut_inds[1]]
    phase = phase[...,cut_inds[0]:cut_inds[1]]

    erpac = ep.fit(phase, power, **fit_args)
    times = epo.times[cut_inds[0]:cut_inds[1]]
    n = phase.shape[1]

    return erpac, times, n

def compare_rho(erpac_a, n_a, erpac_b, n_b, fdr=0.05):
    erpac_a_fish = np.arctanh(erpac_a)
    erpac_b_fish = np.arctanh(erpac_b)
    erpac_delt = erpac_b_fish - erpac_a_fish
    delt_se = np.sqrt(1/(n_a-3) + 1/(n_b-3))
    erpac_z = erpac_delt / delt_se
    erpac_p = norm.sf(abs(erpac_z))*2
    if fdr:
        erpac_p = fdr_correction(erpac_p, alpha=fdr)[1]

    return erpac_z, erpac_p

def tfce_correct(data, tfce_thresh=None):
    if tfce_thresh is None:
        tfce_thresh = dict(start=0, step=0.2)
    pos_data = data.copy()
    pos_data[pos_data<0] = 0
    neg_data = data.copy()
    neg_data[neg_data>0] = 0
    pos_clusts = _find_clusters(pos_data, tfce_thresh)[1].reshape(data.shape)
    neg_clusts = _find_clusters(neg_data, tfce_thresh)[1].reshape(data.shape)
    out_data = np.zeros_like(data) + pos_clusts - neg_clusts
    return out_data

def permute(perm_idx, a_n, epo_inds, phase, power, fit_args, subj_inds=None):
    print("Permutation {} of {}".format(perm_idx, n_perm))
    # if we have group as a factor, we shuffle data only within subjects
    if subj_inds is not None:
        subjs = list(np.unique(subj_inds))
        for subj in subjs:
            subj_inds = subj_inds==subj
            these_epo_inds = epo_inds[subj_inds].copy()
            np.random.shuffle(these_epo_inds)
            epo_inds[subj_inds] = these_epo_inds
        else:
            np.random.shuffle(epo_inds)
    erpac_a = ep.fit(phase[:,epo_inds[:a_n],], power[:,epo_inds[:a_n],], **fit_args)
    erpac_b = ep.fit(phase[:,epo_inds[a_n:],], power[:,epo_inds[a_n:],], **fit_args)
    erpac_z, _ = compare_rho(erpac_a, a_n, erpac_b, len(epo_inds)-a_n,
                             fdr=None)
    erpac_c = tfce_correct(erpac_z)
    return (erpac_c.max(), erpac_c.min())

def do_erpac_perm(epo_a, epo_b, cut, baseline=None, n_perm=1000,
                  fit_args={"mcp":"fdr", "p":0.05, "n_jobs":1,
                            "method":"circular"}):
    data_a = epo_a.get_data()[:,0,] * 1e+6
    data_b = epo_b.get_data()[:,0,] * 1e+6
    data = np.vstack((data_a, data_b))
    epo_inds = np.arange(len(data))
    perm_maxima, perm_minima = np.zeros(n_perm), np.zeros(n_perm)
    a_n = len(epo_a)
    subj_inds = np.hstack((epo_a.metadata["Subj"].values,
                           epo_b.metadata["Subj"].values))

    phase = ep.filter(cond_epo.info["sfreq"], data, ftype="phase", n_jobs=n_jobs)
    power = ep.filter(cond_epo.info["sfreq"], data, ftype="amplitude", n_jobs=n_jobs)

    if baseline:
        base_inds = cond_epo.time_as_index((baseline[0], baseline[1]))
        bl = power[...,base_inds[0]:base_inds[1]]
        bl_mu = bl.mean(axis=-1, keepdims=True)
        bl_std = bl.std(axis=-1, keepdims=True)
        power = (power - bl_mu) / bl_std

    cut_inds = epo.time_as_index((cut[0], cut[1]))
    power = power[...,cut_inds[0]:cut_inds[1]]
    phase = phase[...,cut_inds[0]:cut_inds[1]]
    times = epo.times[cut_inds[0]:cut_inds[1]]

    results = Parallel(n_jobs=1, verbose=10)(delayed(permute)(
                       i, a_n, epo_inds, phase, power, fit_args, subj_inds)
                       for i in range(n_perm))

    return results

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 2
chan = "central"
osc_types = ["SO", "deltO"]
osc_types = ["SO"]
sfreq = 200.
phase_freqs = {"SO":(0.5, 1.25),"deltO":(1.25, 4)}
power_freqs = (5, 25)
conds = ["eig", "fix"]
conds = ["fix"]
durs = ["30s", "2m", "5m"]
osc_cuts = {"SO":(-1.5,1.5),"deltO":(-.75,.75)}
baseline = (-2.35, -1.5)
#baseline = None
method = "wavelet"
exclude = ["002", "003", "028", "007", "051"]
p = 0.05
n_perm = 1000
tfce_thresh = dict(start=0, step=0.2)
recalc = True

f_amp = np.linspace(power_freqs[0],power_freqs[1],50)
epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
for excl in exclude:
    epo = epo["Subj!='{}'".format(excl)]
epo.resample(sfreq, n_jobs="cuda")

epos = []
dfs = []
for osc in osc_types:
    osc_epo = epo["OscType == '{}'".format(osc)]
    pf = phase_freqs[osc]
    osc_cut = osc_cuts[osc]
    ep = ERPAC(f_pha=pf, f_amp=f_amp, dcomplex=method)
    sham_epo = osc_epo["StimType == 'sham'"]
    sham_erpac, times, sham_n = do_erpac(ep, sham_epo, osc_cut, baseline=baseline)
    erpacs = []
    ns = []
    for cond in conds:
        cond_epo = osc_epo["StimType == '{}'".format(cond)]
        erpac, times, n = do_erpac(ep, cond_epo, osc_cut, baseline=baseline)
        erpacs.append(erpac)
        ns.append(n)

        erpac_z, erpac_p = compare_rho(sham_erpac, sham_n, erpac, n, fdr=None)
        erpac_z = erpac_z.squeeze()
        erpac_c = _find_clusters(erpac_z, threshold=tfce_thresh)
        erpac_c = np.reshape(erpac_c[1], erpac_z.shape)

        #ep.pacplot(erpac_c, times, ep.yvec)

        if recalc:
            results = do_erpac_perm(sham_epo, cond_epo, osc_cut, baseline=baseline)
            results = np.array(results)
            np.save("{}{}_erpac_perm.npy".format(proc_dir, cond), results)
        else:
            results = np.load("{}{}_erpac_perm.npy".format(proc_dir, cond))

        pos_thresh_val = np.quantile(results, 1-p)
        erpac_pos_mask = erpac_c > pos_thresh_val
        neg_thresh_val = np.quantile(results, p)
        erpac_neg_mask = erpac_c < neg_thresh_val
        erpac_mask = ~(erpac_pos_mask | erpac_neg_mask)

        # make mne tfr template for plotting
        e = epo[0].crop(tmin=osc_cut[0], tmax=osc_cut[1]-1/sfreq)
        tfr = tfr_morlet(e, f_amp[:-1], n_cycles=5, average=False, return_itc=False)
        tfr = tfr.average()

        fig, ax = plt.subplots(figsize=(24,19.2))
        # ep.pacplot(erpac.squeeze(), times, ep.yvec,
        #            pvalues=ep.pvalues.squeeze(), p=p)
        # ep.pacplot(erpac_mask.squeeze(), times, ep.yvec)
        tfr.data[0,:,:] = erpac_z.squeeze()
        tfr.plot(mask=erpac_mask, mask_style="contour", cmap="inferno",
                 vmin=-3, vmax=3, axes=ax, picks="central")
        # plt.title("{} ERPAC {}, phase at {}-{}Hz ({} transform)".format(osc,
        #                                                                 cond,
        #                                                                 pf[0],
        #                                                                 pf[1],
        #                                                                 method))
        plt.ylabel("Frequency (Hz)", fontdict={"size":36})
        plt.xlabel("Time (s)", fontdict={"size":36})

        cut_inds = epo.time_as_index((osc_cut[0], osc_cut[1]))
        evo = cond_epo.average().data[0,cut_inds[0]:cut_inds[1]]
        evo = (evo - evo.min())/(evo.max()-evo.min())
        evo = evo*7 + 11

        plt.plot(times, evo, linewidth=10, color="gray", alpha=0.8)
        if cond == "fix":
            cond_txt = "Fixed"
        elif cond=="eig":
            cond_txt = "Eigen"
        plt.suptitle("ERPAC for {}, normalised difference: {} - Sham".format(osc, cond_txt))
        plt.savefig("../images/ERPAC_{}_{}_{}.png".format(osc, cond, method))
        plt.savefig("../images/ERPAC_{}_{}_{}.svg".format(osc, cond, method))

        # # get clusters of significant points and examine them at maxima
        # # (p-value minima)
        # p_vals = ep.pvalues.squeeze()
        # p_thr = (p_vals<0.035).astype(int)
        # clusters, sums = _find_clusters(p_thr, 0.99)
        # for clust in clusters:
        #     clu = np.reshape(clust, p_thr.shape)
        #     inds = np.where(clu)
        #     minpoint = (inds[0][np.argmin(p_vals[inds])],
        #                 inds[1][np.argmin(p_vals[inds])])
        #     pt_power = power[minpoint[0], :, minpoint[1]]
        #     pt_phase = phase[0, :, minpoint[1]]
        #
        #     phase_range = np.linspace(-np.pi, np.pi, 37)
        #     binned_phases = np.digitize(pt_phase, phase_range)
        #     phase_bins = [phase_range[x] + (phase_range[x+1]-phase_range[x])/2
        #                   for x in range(len(phase_range)-1)]
        #     bin_avgs = np.zeros_like(phase_bins)
        #     for pb_idx, pb in enumerate(phase_bins):
        #         bin_avgs[pb_idx] = pt_power[binned_phases==pb_idx+1].mean()
        #
        #
        #     plt.figure()
        #     ax = plt.subplot(1,1,1,projection="polar")
        #     #ax.scatter(pt_phase, pt_power, alpha=0.1)
        #     #ax.set_ylim((0,300))
        #     plt.bar(phase_bins, bin_avgs)
        #     plt.title("Phase/Power at {:.2f}s, {:.1f}Hz".format(times[minpoint[1]],
        #                                                     ep.yvec[minpoint[0]]))
        #
        #     print("Mean at {:.2f}s, {:.1f}Hz: {:.2f}".format(times[minpoint[1]],
        #                                              ep.yvec[minpoint[0]],
        #                                              pt_power.mean()))
            #breakpoint()
