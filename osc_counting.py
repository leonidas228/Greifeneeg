import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

epo = mne.read_epochs("{}grand_central_finfo-epo.fif".format(proc_dir))
df = epo.metadata.copy()
df = df.query("PrePost=='Post'")
subjs = list(df["Subj"].unique())
subjs.sort()

df_dict = {"Subj":[], "Dur":[], "Stim":[], "OscType":[], "Total":[]}
for subj in subjs:
    sub_df = df.query("Subj=='{}'".format(subj))
    for dur in ["30s", "2m", "5m"]:
        for stim in ["sham", "fix", "eig"]:
            cond = stim+dur
            cond_df = sub_df.query("Cond=='{}'".format(cond))
            for osc in ["SO", "deltO"]:
                osc_df = cond_df.query("OscType=='{}'".format(osc))
                total = len(osc_df)
                if total:
                    df_dict["Subj"].append(subj)
                    df_dict["Dur"].append(dur)
                    df_dict["Stim"].append(stim)
                    df_dict["OscType"].append(osc)
                    df_dict["Total"].append(total)
total_df = pd.DataFrame.from_dict(df_dict)

fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(x="Stim", y="Total", hue="Dur", data=total_df.query("OscType=='SO'"),
            order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"], ax=ax)
plt.ylim((0,40))
plt.title("Number of SO")
plt.savefig("{}SO_number.tif".format(proc_dir))

fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(x="Stim", y="Total", hue="Dur", data=total_df.query("OscType=='deltO'"),
            order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"], ax=ax)
plt.ylim((0,40))
plt.title("Number of deltO")
plt.savefig("{}deltO_number.tif".format(proc_dir))

fig, axes = plt.subplots(6,6, figsize=(38.4,21.6))
axes = [ax for axe in axes for ax in axe]
for idx, (ax, subj) in enumerate(zip(axes, subjs)):
    this_df = total_df.query("Subj=='{}'".format(subj))
    sns.barplot(x="Stim", y="Total", hue="Dur", data=this_df.query("OscType=='SO'"),
                order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"], ax=ax)
    ax.get_legend().remove()
    ax.set_ylim((0,55))
    ax.set_title(subj)
if idx < len(axes):
    for idx in range(idx+1,len(axes)):
        axes[idx].axis("off")
plt.suptitle("SO by subject")
plt.tight_layout()
