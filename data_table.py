import mne
from os import listdir
import re
import numpy as np
from os.path import join
from mne.time_frequency import tfr_morlet
from scipy.integrate import simps
from tensorpac import EventRelatedPac as ERPAC
import pandas as pd
from scipy.stats import norm

def do_erpac(ep, epo, cut, baseline=None, fit_args={"mcp":"fdr", "p":0.05,
                                                    "n_jobs":8,
                                                    "method":"circular",
                                                    "n_perm":1000}):
    data = epo.get_data()[:,0,] * 1e+6
    phase = ep.filter(epo.info["sfreq"], data, ftype="phase",
                      n_jobs=fit_args["n_jobs"])
    power = ep.filter(epo.info["sfreq"], data, ftype="amplitude",
                      n_jobs=fit_args["n_jobs"])

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

def compare_rho(erpac_a, n_a, erpac_b, n_b):
    erpac_a_fish = np.arctanh(erpac_a)
    erpac_b_fish = np.arctanh(erpac_b)
    erpac_delt = erpac_b_fish - erpac_a_fish
    delt_se = np.sqrt(1/(n_a-3) + 1/(n_b-3))
    erpac_z = erpac_delt / delt_se
    erpac_p = norm.sf(abs(erpac_z))*2

    return erpac_z, erpac_p

root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
conds = ["fix5m", "fix2m", "fix30s", "sham30s", "sham2m", "sham5m"]
filelist = listdir(proc_dir)
chan = "central"

columns = ["EpoN", "SOPower", "SOPowerRel", "TFR", "ERPACLow", "ERPACHigh",
           "ERPACAll"]
df_dict = {"Subject":[]}
for cond in ["sham", "fix"]:
    for col in columns:
        df_dict[f"{cond}_{col}"] = []
cont_conds = ["ERPACLowContrast", "ERPACHighContrast", "ERPACAllContrast"]
for cc in cont_conds:
    df_dict[cc] = []
    df_dict[cc+"_p"] = []

# get all subjs
subjs = []
for filename in filelist:
    this_match = re.match("d_NAP_(\d{3})_.*-epo.fif",filename)
    if this_match:
        subj = this_match.group(1)
        subjs.append(subj)
subjs = list(np.unique(subjs))

# check that each subj has all conditions
new_subjs = []
for subj in subjs:
    t = [f"d_NAP_{subj}_{cond}_{chan}_SO-epo.fif" in filelist for cond in conds]
    if all(t):
        new_subjs.append(subj)
subjs = new_subjs

for subj in subjs:
    fix_epos, sham_epos = [], []
    for cond in conds:
        epo = mne.read_epochs(join(proc_dir,
                                   f"d_NAP_{subj}_{cond}_{chan}_SO-epo.fif"))
        epo.pick_channels([chan])
        if "fix" in epo.filename:
            fix_epos.append(epo)
        elif "sham" in epo.filename:
            sham_epos.append(epo)
        else:
            raise ValueError("Don't know what condition this is")
    fix_epo = mne.concatenate_epochs(fix_epos)
    sham_epo = mne.concatenate_epochs(sham_epos)

    df_dict["Subject"].append(subj)
    for cond, epo in zip(["fix", "sham"], [fix_epo, sham_epo]):
        df_dict[f"{cond}_EpoN"].append(len(epo))
        # tfr
        tfr = tfr_morlet(epo, np.arange(12,19), 5, return_itc=False, n_jobs=8)
        tfr.crop(tmin=-2.35, tmax=2.35)
        tfr.apply_baseline((-2.35, -1.5), mode="zscore")
        # extract tfwin
        tfr.crop(.09, 0.18, 15, 18)
        tfr_data = tfr.data.mean()
        df_dict[f"{cond}_TFR"].append(tfr_data)

        # SO Power
        psd = epo.compute_psd(method="welch", n_fft=2048, fmax=40).average()
        freq_res = psd.freqs[1] - psd.freqs[0]
        total_power = simps(psd.get_data().squeeze()*1e12, dx=freq_res)
        so_power = simps(psd.get_data(fmin=0.5, fmax=1.25).squeeze()*1e12,
                         dx=freq_res)
        so_power_rel = so_power / total_power
        df_dict[f"{cond}_SOPower"].append(so_power)
        df_dict[f"{cond}_SOPowerRel"].append(so_power_rel)

        # erpac
        # 12-15Hz, .25-.65
        ep = ERPAC(f_pha=(0.5, 1.25), f_amp=(12, 15), dcomplex="wavelet")
        erpac, times, n = do_erpac(ep, epo, (-1.5, 1.5), baseline=(-2.35, -1.5))
        time_inds = (np.where(times==.25)[0][0], np.where(times==.65)[0][0])
        erpac_lowband = erpac[...,time_inds[0]:time_inds[1]].mean()
        df_dict[f"{cond}_ERPACLow"].append(erpac_lowband)

        # 15-18Hz, .15-.38
        ep = ERPAC(f_pha=(0.5, 1.25), f_amp=(15, 18), dcomplex="wavelet")
        erpac, times, n = do_erpac(ep, epo, (-1.5, 1.5), baseline=(-2.35, -1.5))
        time_inds = (np.where(times==.15)[0][0], np.where(times==.38)[0][0])
        erpac_highband = erpac[...,time_inds[0]:time_inds[1]].mean()
        df_dict[f"{cond}_ERPACHigh"].append(erpac_highband)

        # 12-18Hz, .15-.65
        ep = ERPAC(f_pha=(0.5, 1.25), f_amp=(12, 18), dcomplex="wavelet")
        erpac, times, n = do_erpac(ep, epo, (-1.5, 1.5), baseline=(-2.35, -1.5))
        time_inds = (np.where(times==.15)[0][0], np.where(times==.65)[0][0])
        erpac_all = erpac[...,time_inds[0]:time_inds[1]].mean()
        df_dict[f"{cond}_ERPACAll"].append(erpac_all)

    # ERPAC contrasts
    e_r, e_p = compare_rho(df_dict["sham_ERPACLow"][-1], df_dict["sham_EpoN"][-1],
                           df_dict["fix_ERPACLow"][-1], df_dict["fix_EpoN"][-1])
    df_dict["ERPACLowContrast"].append(e_r)
    df_dict["ERPACLowContrast_p"].append(e_p)
    e_r, e_p = compare_rho(df_dict["sham_ERPACHigh"][-1], df_dict["sham_EpoN"][-1],
                           df_dict["fix_ERPACHigh"][-1], df_dict["fix_EpoN"][-1])
    df_dict["ERPACHighContrast"].append(e_r)
    df_dict["ERPACHighContrast_p"].append(e_p)
    e_r, e_p = compare_rho(df_dict["sham_ERPACAll"][-1], df_dict["sham_EpoN"][-1],
                           df_dict["fix_ERPACAll"][-1], df_dict["fix_EpoN"][-1])
    df_dict["ERPACAllContrast"].append(e_r)
    df_dict["ERPACAllContrast_p"].append(e_p)

df = pd.DataFrame.from_dict(df_dict)
df.to_pickle(join(proc_dir, "dataframe.pickle"))
df.to_csv(join(proc_dir, "dataframe.csv"))
