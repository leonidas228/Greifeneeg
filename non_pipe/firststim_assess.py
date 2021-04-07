import mne
from os import listdir
import re
from os.path import isdir
import pandas as pd
import seaborn as sns
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
img_dir = "../images/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s", "sham30s", "sham2m", "sham5m"]
filelist = listdir(proc_dir)
excludes = []

df_dict = {"Subj":[], "StimType":[], "Dur":[], "Cond":[], "Sync":[], "Onset":[]}

try:
    df = pd.read_pickle("{}onset_df.pickle".format(proc_dir))
except:
    for filename in filelist:
        this_match = re.match("af_NAP_(\d{3})_(.*)-raw.fif",filename)
        if this_match:
            subj, cond = this_match.group(1), this_match.group(2)
            if cond not in conds or "{}_{}".format(subj,cond) in excludes:
                continue
            raw = mne.io.Raw(proc_dir+filename,preload=True)
            raws = []
            for annot in raw.annotations:
                match = re.match("(.*)_Stimulation (\d)", annot["description"])
                if match:
                    stim_pos, stim_idx = match.group(1), match.group(2)
                    if stim_pos != "BAD" or stim_idx != "0":
                        continue

                    df_dict["Subj"].append(subj)
                    if int(subj) < 31:
                        df_dict["Sync"].append("async")
                    else:
                        df_dict["Sync"].append("sync")

                    df_dict["Cond"].append(cond)
                    if "eig" in cond:
                        df_dict["StimType"].append("eig")
                    elif "fix" in cond:
                        df_dict["StimType"].append("fix")
                    elif "sham" in cond:
                        df_dict["StimType"].append("sham")
                    if "30s" in cond:
                        df_dict["Dur"].append("30s")
                    elif "2m" in cond:
                        df_dict["Dur"].append("2m")
                    elif "5m" in cond:
                        df_dict["Dur"].append("5m")

                    df_dict["Onset"].append(annot["onset"])
    df = pd.DataFrame.from_dict(df_dict)
    df.to_pickle("{}onset_df.pickle".format(proc_dir))

df["Onset"] = df["Onset"] / 60

this_df = df.query("StimType!='sham'")
plt.figure(figsize=(38.4,21.6))
sns.histplot(data=this_df, x="Onset", hue="Sync", bins=30)
plt.title("Histogram of 1st stimulation onsets (m), Eigen and fixed frequency")
plt.xlabel("Onset (m)")
plt.savefig("{}first_stim_onsets_hist.tif".format(img_dir))

plt.figure(figsize=(38.4,21.6))
sns.barplot(data=this_df, x="Sync", y="Onset")
plt.title("Mean of 1st stimulation onsets (m), Eigen and fixed frequency")
plt.ylabel("Onset (m)")
plt.savefig("{}first_stim_onsets_mean.tif".format(img_dir))

this_df = df.query("StimType=='eig'")
plt.figure(figsize=(38.4,21.6))
sns.histplot(data=this_df, x="Onset", hue="Sync", bins=30)
plt.title("Histogram of 1st stimulation onsets (m), Eigenfrequency only")
plt.xlabel("Onset (m)")
plt.savefig("{}first_stim_onsets_eigen_hist.tif".format(img_dir))

plt.figure(figsize=(38.4,21.6))
sns.barplot(data=this_df, x="Sync", y="Onset")
plt.title("Mean of 1st stimulation onsets (m), Eigenfrequency only")
plt.ylabel("Onset (m)")
plt.savefig("{}first_stim_onsets_eigen_mean.tif".format(img_dir))

this_df = df.query("StimType=='fix'")
plt.figure(figsize=(38.4,21.6))
sns.histplot(data=this_df, x="Onset", hue="Sync", bins=30)
plt.title("Histogram of 1st stimulation onsets (m), Fixed frequency only")
plt.xlabel("Onset (m)")
plt.savefig("{}first_stim_onsets_fixed_hist.tif".format(img_dir))

plt.figure(figsize=(38.4,21.6))
sns.barplot(data=this_df, x="Sync", y="Onset")
plt.title("Mean of 1st stimulation onsets (m), Fixed frequency only")
plt.ylabel("Onset (m)")
plt.savefig("{}first_stim_onsets_fixed_mean.tif".format(img_dir))
