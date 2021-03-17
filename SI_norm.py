import mne
import numpy as np
import pandas as pd
from scipy.stats import circmean
from os.path import isdir
import statsmodels.formula.api as smf
import seaborn as sns
from circular_hist import circ_hist_norm
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

def r_vector(rad):
    x_bar, y_bar = np.cos(rad).mean(), np.sin(rad).mean()
    r_mean = circmean(rad, low=-np.pi, high=np.pi)
    r_norm = np.linalg.norm((x_bar, y_bar))

    return r_mean, r_norm

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

epo = mne.read_epochs("{}grand_central_finfo_SI-epo.fif".format(proc_dir))
df = epo.metadata

subjs = list(df["Subj"].unique())
subjs.sort()
conds = ["sham", "eig", "fix"]
durs = ["30s", "2m", "5m"]
osc_types = ["SO", "deltO"]
syncs = ["sync", "async", "all"]
syncs = ["all"]
method = "hilbert"

SI_df_dict = {"Subj":[], "Sync":[], "OscType":[], "Cond":[], "StimType":[],
              "Dur":[], "SI_norm":[], "SM_norm":[], "SI_mean":[], "SM_mean":[]}
for subj in subjs:
    for osc in osc_types:
        for cond in conds:
            for dur in durs:
                query_str = "Subj=='{}' and OscType=='{}' and StimType=='{}' and Dur=='{}'".format(subj, osc, cond, dur)
                this_df = df.query(query_str)
                if len(this_df) > 10:
                    SIs = this_df["SI"].values
                    SMs = this_df["Spind_Max"].values
                    SI_mean, SI_r_norm = r_vector(SIs)
                    SM_mean, SM_r_norm = r_vector(SMs)
                    SI_df_dict["Subj"].append(subj)
                    if int(subj) < 31:
                        SI_df_dict["Sync"].append("async")
                    else:
                        SI_df_dict["Sync"].append("sync")
                    SI_df_dict["OscType"].append(osc)
                    SI_df_dict["Cond"].append(cond+dur)
                    SI_df_dict["StimType"].append(cond)
                    SI_df_dict["Dur"].append(dur)
                    SI_df_dict["SI_norm"].append(SI_r_norm)
                    SI_df_dict["SM_norm"].append(SM_r_norm)
                    SI_df_dict["SI_mean"].append(SI_mean)
                    SI_df_dict["SM_mean"].append(SM_mean)
SI_df = pd.DataFrame.from_dict(SI_df_dict)

for sync in syncs:
    if sync != "all":
        sync_df = df.query("Sync=='{}'".format(sync))
        sync_SI_df = SI_df.query("Sync=='{}'".format(sync))
        sync_string = ", {} only".format(sync)
    else:
        sync_df = df.copy()
        sync_SI_df = SI_df.copy()
        sync_string = ""
    for osc in osc_types:
        fig, axes = plt.subplots(len(conds),len(durs),figsize=(38.4,21.6),
                                 subplot_kw={"projection":"polar"})
        for dur_idx, dur in enumerate(durs):
            for cond_idx, cond in enumerate(conds):
                query_str = "OscType=='{}' and StimType=='{}' and Dur=='{}'".format(osc, cond, dur)
                this_df = sync_df.query(query_str)
                this_SI_df = sync_SI_df.query(query_str)
                subj_spinds = this_SI_df["SM_mean"].values
                subj_mean, subj_r_norm = r_vector(subj_spinds)
                mean, r_norm = r_vector(this_df["Spind_Max"].values)
                vecs = [[(subj_mean, subj_r_norm), {"color":"red","linewidth":4}],
                        [(mean, r_norm), {"color":"blue","linewidth":4}]]
                circ_hist_norm(axes[dur_idx,cond_idx], this_df["Spind_Max"].values,
                               points=subj_spinds, vecs=vecs, alpha=0.3,
                               points_col="red", bins=48)
                axes[dur_idx,cond_idx].set_title("{} {}".format(cond, dur))
        plt.suptitle("Spindle Peak on {} phase {} ({} transform)".format(osc, sync_string, method))
        plt.tight_layout()
        plt.savefig("../images/polar_hist_{}_{}_{}".format(osc, sync, method))

d = SI_df.query("OscType=='SO'")
# sns.catplot(data=d, x="StimType", hue="Dur", y="SM_norm", kind="bar", col="Sync")
# plt.ylabel("Resultant Vector")
# plt.suptitle("Slow Oscillations ({} transform)".format(method))
# plt.savefig("../images/resvec_bar_SO_sync")
fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(data=d, x="StimType", hue="Dur", y="SM_norm", ax=ax)
plt.ylabel("Resultant Vector")
plt.suptitle("Slow Oscillations ({} transform)".format(method))
plt.savefig("../images/resvec_bar_SO_all_{}".format(method))

formula = "SM_norm ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))"
mod = smf.mixedlm(formula, data=d, groups=d["Sync"])
mf = mod.fit()
print(mf.summary())

d = SI_df.query("OscType=='deltO'")
# sns.catplot(data=d, x="StimType", hue="Dur", y="SM_norm", kind="bar", col="Sync")
# plt.ylabel("Resultant Vector")
# plt.suptitle("Delta Oscillations ({} transform)".format(method))
# plt.savefig("../images/resvec_bar_deltO_sync")
fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(data=d, x="StimType", hue="Dur", y="SM_norm", ax=ax)
plt.ylabel("Resultant Vector")
plt.suptitle("Delta Oscillations ({} transform)".format(method))
plt.savefig("../images/resvec_bar_deltO_all_{}".format(method))

formula = "SM_norm ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))"
mod = smf.mixedlm(formula, data=d, groups=d["Sync"])
mf = mod.fit()
print(mf.summary())
