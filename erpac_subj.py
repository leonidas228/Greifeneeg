import mne
from mne.stats.cluster_level import _find_clusters
from joblib import Parallel, delayed
from mne.stats import fdr_correction
from tensorpac import EventRelatedPac as ERPAC
#from tensorpac import PreferredPhase as PP
import pandas as pd
import numpy as np
from scipy.stats import norm, sem
from os.path import isdir
import matplotlib.pyplot as plt
import pickle
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

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
phase_freqs = [(0.5, 1.25),(1.25, 4)]
power_freqs = (13, 17)
conds = ["eig", "fix"]
durs = ["30s", "2m", "5m"]
osc_cuts = [(-1.5,1.5),(-.75,.75)]
baseline = (-2.35, -1.5)
method = "wavelet"
exclude = ["002", "003", "028"]
p = 0.05
tfce_thresh = dict(start=0, step=0.2)
tfce_thresh = None

f_amp = np.linspace(power_freqs[0],power_freqs[1],50)
epo = mne.read_epochs("{}grand_{}_finfo-epo.fif".format(proc_dir, chan),
                      preload=True)
for excl in exclude:
    epo = epo["Subj!='{}'".format(excl)]
epo.resample(sfreq, n_jobs="cuda")

subjs = list(np.sort(epo.metadata["Subj"].unique()))

erpacs = {osc:{cond:[] for cond in conds} for osc in osc_types}

for osc, osc_cut, pf in zip(osc_types, osc_cuts, phase_freqs):
    osc_epo = epo["OscType == '{}'".format(osc)]
    ep = ERPAC(f_pha=pf, f_amp=power_freqs, dcomplex=method)
    for subj in subjs:
        subj_epo = osc_epo["Subj=='{}'".format(subj)]
        cond_epo = subj_epo["StimType == 'sham'"]
        sham_erpac, times, sham_n = do_erpac(ep, cond_epo, osc_cut,
                                   baseline=baseline)
        for cond in conds:
            cond_epo = subj_epo["StimType == '{}'".format(cond)]
            cond_erpac, times, cond_n = do_erpac(ep, cond_epo, osc_cut,
                                       baseline=baseline)
            delta = compare_rho(sham_erpac, sham_n, cond_erpac, cond_n, fdr=None)
            erpacs[osc][cond].append(delta)


    # graph
    #sham = np.array(erpacs[osc]["sham"]).squeeze()
    eig = np.array(erpacs[osc]["eig"]).squeeze()
    fix = np.array(erpacs[osc]["fix"]).squeeze()

    #sham_mean, sham_sem = sham.mean(axis=0), sem(sham, axis=0)
    # eig_mean, eig_sem = eig.mean(axis=0), sem(eig, axis=0)
    # fix_mean, fix_sem = fix.mean(axis=0), sem(fix, axis=0)
    #
    # plt.figure()
    # plt.plot(times, sham_mean, color="blue", linewidth=4)
    # plt.fill_between(times, sham_mean + sham_sem*1.96,
    #                  sham_mean - sham_sem*1.96, color="blue", alpha=.1)
    # plt.plot(times, eig_mean, color="red", linewidth=4)
    # plt.fill_between(times, eig_mean + eig_sem*1.96,
    #                  eig_mean - eig_sem*1.96, color="red", alpha=.1)
    #
    # plt.figure()
    # plt.plot(times, sham_mean, color="blue", linewidth=4)
    # plt.fill_between(times, sham_mean + sham_sem*1.96,
    #                  sham_mean - sham_sem*1.96, color="blue", alpha=.1)
    # plt.plot(times, fix_mean, color="green", linewidth=4)
    # plt.fill_between(times, fix_mean + fix_sem*1.96,
    #                  fix_mean - fix_sem*1.96, color="green", alpha=.1)
