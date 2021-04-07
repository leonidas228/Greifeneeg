import mne
from mne.time_frequency import psd_multitaper
import numpy as np
from os.path import isdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"

epo = mne.read_epochs(proc_dir+"grand_central-epo.fif", preload=True)
df = epo.metadata.copy()
subjs = list(df["Subj"].unique())
stim_freqs = np.empty(len(df))
avg_SO_freq = np.empty(len(df))

graph_df = {"Subj":[],"Cond":[], "OscFreq":[]}

for subj in subjs:
    stim_raw = mne.io.Raw("{}bad_caf_NAP_{}_eig30s-raw.fif".format(proc_dir, subj))
    psds, freqs = psd_multitaper(stim_raw, fmin=0.5, fmax=1, n_jobs=8, bandwidth=0.01)
    psd = psds.mean(axis=0)
    eig_freq = freqs[np.argmax(psd)]

    inds = np.array(((df["Subj"] == subj) & (df["StimType"]=='eig')))
    stim_freqs[inds] = eig_freq
    inds = np.array(((df["Subj"] == subj) & (df["StimType"]=='fix')))
    stim_freqs[inds] = 0.75
    inds = np.array(((df["Subj"] == subj) & (df["StimType"]=='sham')))
    stim_freqs[inds] = np.nan

    inds = np.array(((df["Subj"] == subj) & (df["OscType"]=='SO') & (df["StimType"]=='sham')))
    SO_freq_mean = df.iloc[inds]["OscFreq"].mean()
    inds = np.array(((df["Subj"] == subj) & (df["OscType"]=='deltO') & (df["StimType"]=='sham')))
    deltO_freq_mean = df.iloc[inds]["OscFreq"].mean()
    inds = np.array(df["Subj"] == subj)
    avg_SO_freq[inds] = SO_freq_mean


    graph_df["Subj"].append(subj)
    graph_df["Cond"].append("Stimulated")
    graph_df["OscFreq"].append(eig_freq)

df["StimFreq"] = stim_freqs
df["AvgSOFreq"] = avg_SO_freq
epo.metadata = df
epo.save(proc_dir+"grand_central_finfo-epo.fif", overwrite=True)

# SO_df = df.query("OscType=='SO'")
# subjs = list(SO_df["Subj"].sort_values().unique())
# for subj in subjs:
#     subj_df = SO_df.query("Subj=='{}'".format(subj))
#     for cond in ['fix30s', 'eig30s', 'eig5m', 'fix5m', 'fix2m']:
#         this_df = subj_df.query("Cond=='{}' and PrePost=='Pre'".format(cond))
#         oscfreqs = list(this_df["OscFreq"].values)
#         graph_df["Subj"].extend([subj]*len(oscfreqs))
#         graph_df["Cond"].extend(["Pre_"+cond]*len(oscfreqs))
#         graph_df["OscFreq"].extend(oscfreqs)
#     this_df = subj_df.query("Cond=='sham30s'".format(cond))
#     oscfreqs = list(this_df["OscFreq"].values)
#     graph_df["Subj"].extend([subj]*len(oscfreqs))
#     graph_df["Cond"].extend(["FullSham"]*len(oscfreqs))
#     graph_df["OscFreq"].extend(oscfreqs)
#
# gdf = pd.DataFrame.from_dict(graph_df)
# fig, axes = plt.subplots(7,5,figsize=(38.4,21.6))
# axes = [ax for axe in axes for ax in axe]
# for idx, subj in enumerate(subjs):
#     this_gdf = gdf.query("Subj=='{}'".format(subj))
#     sns.barplot(data=this_gdf, x="Cond", y="OscFreq", ax=axes[idx],
#                 order=["Stimulated", "FullSham", 'Pre_fix30s', 'Pre_eig30s',
#                        'Pre_eig5m', 'Pre_fix5m', 'Pre_fix2m'])
#     axes[idx].set_ylim((0,1.25))
#     axes[idx].set_title(subj)
# plt.tight_layout()
