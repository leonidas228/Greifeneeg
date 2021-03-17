import pickle
import numpy as np
from os.path import isdir
from os import listdir
import pandas as pd
import re
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

def plot_histos(df, title=None, filename=None):

    boot_num = df["boot_num"]
    bins = np.array(list(df["Bins"].values))
    bins = np.average(bins, axis=0, weights=boot_num)
    SO_counts = np.array(list(df["SO_counts"].values))
    SO_counts = np.average(SO_counts, axis=0, weights=boot_num)
    deltO_counts = np.array(list(df["deltO_counts"].values))
    deltO_counts = np.average(deltO_counts, axis=0, weights=boot_num)
    free_counts = np.array(list(df["free_counts"].values))
    free_counts = np.average(free_counts, axis=0, weights=boot_num)

    fig, axes = plt.subplots(2, 3, figsize=(38.4,21.6))
    axes[0][0].bar(bins, free_counts, color="blue")
    axes[0][0].set_title("Free spindle power")
    axes[0][0].set_ylabel("Percent of occurrence")

    axes[0][1].bar(bins, SO_counts, color="green")
    axes[0][1].set_title("SO spindle power")

    axes[0][2].bar(bins, deltO_counts, color="red")
    axes[0][2].set_title("deltO spindle power")

    axes[1][0].bar(bins, free_counts, color="blue", alpha=0.2)
    axes[1][0].bar(bins, SO_counts, color="green", alpha=0.2)
    axes[1][0].set_title("Free and SO spindle power")
    axes[1][0].set_ylabel("Percent of occurrence")
    axes[1][0].set_xlabel("Log Power")

    axes[1][1].bar(bins, free_counts, color="blue", alpha=0.2)
    axes[1][1].bar(bins, deltO_counts, color="red", alpha=0.2)
    axes[1][1].set_title("Free and deltO spindle power")
    axes[1][1].set_xlabel("Log Power")

    axes[1][2].bar(bins, SO_counts, color="green", alpha=0.2)
    axes[1][2].bar(bins, deltO_counts, color="red", alpha=0.2)
    axes[1][2].set_title("SO and deltO spindle power")
    axes[1][2].set_xlabel("Log Power")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if filename:
        fig.savefig(filename)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

perm_n = 1000
conds = ["sham", "fix", "eig"]

filelist = listdir(proc_dir)

df_dict = {"Subj":[], "Cond":[], "StimType":[], "Dur":[], "Bins":[], "SO_counts":[],
           "deltO_counts":[], "free_counts":[], "boot_num":[]}
for filename in filelist:
    # load files, merge SO and deltO annotations
    match_str = "spindle_distros_(.*)_(.*)_(.*).pickle"
    this_match = re.match(match_str, filename)
    if not this_match:
        continue
    subj, cond, chan = (this_match.group(1), this_match.group(2),
                        this_match.group(3))
    with open(proc_dir+filename, "rb") as f:
        histos = pickle.load(f)

        if "fix" in cond:
            stim_type = "fix"
        elif "eig" in cond:
            stim_type = "eig"
        else:
            stim_type = "sham"

        if "30s" in cond:
            dur = "30s"
        elif "2m" in cond:
            dur = "2m"
        else:
            dur = "5m"

        df_dict["Subj"].append(subj)
        df_dict["Cond"].append(cond)
        df_dict["StimType"].append(stim_type)
        df_dict["Dur"].append(dur)
        df_dict["Bins"].append(histos["bin_edges"])
        df_dict["SO_counts"].append(histos["SO_counts"])
        df_dict["deltO_counts"].append(histos["deltO_counts"])
        df_dict["free_counts"].append(histos["free_counts"])
        df_dict["boot_num"].append(histos["boot_num"])

df = pd.DataFrame.from_dict(df_dict)

# all histos
plot_histos(df, "All conditions", "../images/spindle_histos_all.png")
for cond in conds:
    this_df = df.query("StimType=='{}'".format(cond))
    plot_histos(this_df, cond, "../images/spindle_histos_{}.png".format(cond))
