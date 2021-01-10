import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

df = pd.read_pickle("{}grand_df.pickle".format(proc_dir))
df = df.query("PrePost=='Post'")
subjs = np.unique(df["Subj"].values)

df_dict = {"Subj":[], "Dur":[], "Stim":[], "OscType":[], "Total":[]}
for subj in subjs:
    sub_df = df.query("Subj=='{}'".format(subj))
    for dur in ["30s", "2m", "5m"]:
        for stim in ["sham", "fix", "eig"]:
            cond = stim+dur
            cond_df = sub_df.query("Cond=='{}'".format(cond))
            for osc in ["SO", "deltO"]:
                osc_df = cond_df.query("OscType=='{}'".format(osc))
                total = osc_df["Number"].values.sum()
                df_dict["Subj"].append(subj)
                df_dict["Dur"].append(dur)
                df_dict["Stim"].append(stim)
                df_dict["OscType"].append(osc)
                df_dict["Total"].append(total)
total_df = pd.DataFrame.from_dict(df_dict)

total_df_sync = total_df.copy()
bad_subjs = []
for subj in list(subjs):
    if int(subj) < 31:
        bad_subjs.append(subj)
bad_subjs = list(set(bad_subjs))
for bs in bad_subjs:
    print("Removing subject {}".format(bs))
    total_df_sync = total_df_sync.query("Subj!='{}'".format(bs))

fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(x="Stim", y="Total", hue="Dur", data=total_df.query("OscType=='SO'"),
            order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"], ax=ax)
plt.ylim((0,40))
plt.title("Number of SO\nnon-synchronised included")
plt.savefig("{}SO_number_async.tif".format(proc_dir))

fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(x="Stim", y="Total", hue="Dur", data=total_df.query("OscType=='deltO'"),
            order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"], ax=ax)
plt.ylim((0,40))
plt.title("Number of deltO\nnon-synchronised included")
plt.savefig("{}deltO_number_async.tif".format(proc_dir))

fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(x="Stim", y="Total", hue="Dur", data=total_df_sync.query("OscType=='SO'"),
            order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"], ax=ax)
plt.ylim((0,40))
plt.title("Number of SO\nsynchronised only")
plt.savefig("{}SO_number_sync.tif".format(proc_dir))

fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(x="Stim", y="Total", hue="Dur", data=total_df_sync.query("OscType=='deltO'"),
            order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"], ax=ax)
plt.ylim((0,40))
plt.title("Number of deltO\nsynchronised only")
plt.savefig("{}deltO_number_sync.tif".format(proc_dir))
