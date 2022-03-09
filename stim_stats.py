import mne
from os import listdir
import re
from os.path import isdir
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.ion()
import numpy as np

def rel_starts_plot(df, cat_dist=1, in_cat_dist=0.3, jitter=0.05, alpha=.1,
                    conds = ["Sham", "Fixed", "Eigen"], durs=[30, 120, 300],
                    idx_cols={1:"red", 2:"green", 3:"blue", 4:"cyan"},
                    y_buffer=0.25, marksize=100, xlim=(0,4000),
                    fontsize=38):
    fig, ax = plt.subplots(1,1, figsize=(38.4, 19.2))
    ## calculate y coords
    # get the spacing right within the categories
    in_dist = in_cat_dist * (len(durs)-1)
    mid_point = in_dist / 2
    points = np.linspace(0, in_dist, len(durs)) - mid_point
    yticks = {}
    for dur_idx, dur in enumerate(durs):
        for cond_idx, cond in enumerate(conds):
            y = y_buffer + dur_idx * cat_dist + points[cond_idx]
            last_dot = 0
            for idx, idx_col in idx_cols.items():
                # get the data
                df_q = df.copy().query("Cond=='{}' and Dur=={} and StimId=={}".format(cond, dur, idx))
                data = df_q["TimeRel"].values
                ys = np.ones(len(data)) * y
                ys += np.random.uniform(-jitter/2, jitter/2, ys.shape)
                # now scatter plot
                ax.scatter(data, ys, color=idx_col, alpha=alpha, s=marksize)
                median = np.median(data)
                ax.scatter(median, y, color=idx_col, alpha=1, s=marksize*2,
                           edgecolors="black")
                plt.hlines(y, last_dot, median, color="black", linewidth=1)
                ax.set_xlim(xlim)
                last_dot = median
                yticks["{} {}s".format(cond, dur)] = y
    ax.set_yticks(list(yticks.values()))
    ax.set_yticklabels(list(yticks.keys()), fontsize=fontsize)
    ax.invert_yaxis()
    ax.set_xticks([900, 1800, 2700, 3600])
    ax.set_xticklabels(["15m", "30m", "45m", "60m"], fontsize=fontsize)
    ax.set_xlabel("Minutes", fontsize=fontsize)

    legend_comps = []
    for idx, idx_col in idx_cols.items():
        legend_comps.append(Line2D([0], [0], marker='o', color="white",
                            markerfacecolor=idx_col,
                            label="Stim train {}".format(idx+1),
                            markersize=15))
    ax.legend(handles=legend_comps, fontsize=fontsize)

    ax.set_title("Relative stimulation train timings", fontsize=fontsize)


    return ax



if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s", "sham30s", "sham2m", "sham5m"]
#conds = ["sham30s", "sham2m", "sham5m"]
filelist = listdir(proc_dir)
excludes = ["002", "003", "028", "007", "051"]
calc = False

if calc:
    df_dict = {"Subj":[], "StimId":[], "TimeAbs":[], "TimeRel":[],
               "Cond":[], "Dur":[]}
    for filename in filelist:
        this_match = re.match("af_NAP_(\d{3})_(.*)-raw.fif",filename)
        if this_match:
            subj, cond = this_match.group(1), this_match.group(2)
            if cond not in conds or subj in excludes:
                continue
            raw = mne.io.Raw(proc_dir+filename)
            raws = []
            for annot in raw.annotations:
                match = re.match("(.*)_Stimulation (\d)", annot["description"])
                if match:
                    stim_pos, stim_idx = match.group(1), match.group(2)
                    if stim_pos != "BAD":
                        continue
                    begin, duration = annot["onset"], annot["duration"]
                    if stim_idx == "0":
                        orig_time = begin
                    df_dict["Subj"].append(int(subj))
                    df_dict["StimId"].append(int(stim_idx))
                    df_dict["TimeAbs"].append(begin)
                    df_dict["TimeRel"].append(begin - orig_time)
                    if cond[-2:] == "0s":
                        df_dict["Dur"].append(30)
                    elif cond[-2:] == "2m":
                        df_dict["Dur"].append(120)
                    elif cond[-2:] == "5m":
                        df_dict["Dur"].append(300)
                    else:
                        raise ValueError("Stim duration not recognised.")
                    if cond[:3] == "sha":
                        df_dict["Cond"].append("Sham")
                    elif cond[:3] == "fix":
                        df_dict["Cond"].append("Fixed")
                    elif cond[:3] == "eig":
                        df_dict["Cond"].append("Eigen")
                    else:
                        raise ValueError("Stim type not recognised.")

    df = pd.DataFrame.from_dict(df_dict)
    df.to_pickle("{}stim_times.pickle".format(proc_dir))
else:
    df = pd.read_pickle("{}stim_times.pickle".format(proc_dir))

ax = rel_starts_plot(df)
plt.show()
