import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

plt.ion()

def annot_subset(annotations, match_str):
    delete_inds = []
    for ann_idx, ann in enumerate(annotations):
        match = re.search(match_str, ann["description"])
        if not match:
            delete_inds.append(ann_idx)
    annotations.delete(delete_inds)
    return annotations

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)

channel = "central"
SO_border = [-1, 1]
deltO_border = [-.5,.5]
excludes = ["033_eig30s", "025_eig2m", "044_fix2m", "046_eig2m", "022_eig5m",
            "031_eig5m", "035_eig5m", "046_eig5m", "048_eig5m", "044_eig5m",
            "053_eig5m"]
calc = False

if calc:
    df_dict = {"Subj":[], "Cond":[], "SO_nest":[], "Stim":[], "Dur":[],
               "deltO_nest":[], "SO/deltO_nest":[]}
    for filename in filelist:
        this_match = re.match("NAP_(\d{3})_(.*)_(.*)_(.*)-raw.fif",filename)
        if this_match:
            subj, cond, ort, this_osc = this_match.group(1), this_match.group(2), this_match.group(3), this_match.group(4)
            if ort != channel or this_osc != "SO":
                continue
            if "{}_{}".format(subj, cond) in excludes:
                continue

            if "sham" in cond:
                stim = "sham"
            elif "eig" in cond:
                stim = "eig"
            elif "fix" in cond:
                stim = "fix"
            dur = cond[len(stim):]

            # SO
            filename = "spind_{}_NAP_{}_{}_SO-raw.fif".format(subj,cond,ort)
            raw = mne.io.Raw(proc_dir+filename,preload=True)
            SO_spind_n = 0
            SO_spind_nest_n = 0
            for annot_idx, annot in enumerate(raw.annotations):
                if "Spindle" not in annot["description"]:
                    continue
                SO_spind_n += 1
                onset = annot["onset"]
                if (annot_idx != 0) and (raw.annotations[annot_idx-1]["onset"] - onset) > SO_border[0]:
                    SO_spind_nest_n += 1
                    continue
                if (annot_idx != len(raw.annotations)-1) and (raw.annotations[annot_idx+1]["onset"] - onset) < SO_border[1]:
                    SO_spind_nest_n += 1
            SO_ratio = SO_spind_nest_n/SO_spind_n
            print("{} {}, SO ratio: {}".format(subj, cond, SO_ratio))
            SO_ratio = 1 if SO_ratio == 0 else SO_ratio

            # deltO
            filename = "spind_{}_NAP_{}_{}_deltO-raw.fif".format(subj,cond,ort)
            raw = mne.io.Raw(proc_dir+filename,preload=True)
            deltO_spind_n = 0
            deltO_spind_nest_n = 0
            for annot_idx, annot in enumerate(raw.annotations):
                if "Spindle" not in annot["description"]:
                    continue
                deltO_spind_n += 1
                onset = annot["onset"]
                if (annot_idx != 0) and (raw.annotations[annot_idx-1]["onset"] - onset) > deltO_border[0]:
                    deltO_spind_nest_n += 1
                    continue
                if (annot_idx != len(raw.annotations)-1) and (raw.annotations[annot_idx+1]["onset"] - onset) < deltO_border[1]:
                    deltO_spind_nest_n += 1
            deltO_ratio = deltO_spind_nest_n/deltO_spind_n
            print("{} {}, deltO ratio: {}".format(subj, cond, deltO_ratio))
            deltO_ratio = 1 if deltO_ratio == 0 else deltO_ratio

            print("{} {}, SO/deltO ratio: {}".format(subj, cond, SO_ratio/deltO_ratio))

            df_dict["Subj"].append(subj)
            df_dict["Cond"].append(cond)
            df_dict["Stim"].append(stim)
            df_dict["Dur"].append(dur)
            df_dict["SO_nest"].append(SO_ratio)
            df_dict["deltO_nest"].append(deltO_ratio)
            df_dict["SO/deltO_nest"].append(SO_ratio/deltO_ratio)

            df = pd.DataFrame.from_dict(df_dict)
            df.to_pickle("{}spindle_nesting.pickle".format(proc_dir))
else:
    df = pd.read_pickle("{}spindle_nesting.pickle".format(proc_dir))

fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(x="Stim", y="SO/deltO_nest", hue="Dur", data=df,
            order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"],
            ax=ax)
plt.ylim((0,3.5))
plt.title("Spindle nesting, SO/deltO ratio\nnon-synchronised included")
plt.savefig("nesting_async.tif")

df_sync = df.copy()
subjs = np.unique(df["Subj"].values)
bad_subjs = []
for subj in list(subjs):
    if int(subj) < 31:
        bad_subjs.append(subj)
bad_subjs = list(set(bad_subjs))
for bs in bad_subjs:
    print("Removing subject {}".format(bs))
    df_sync = df_sync.query("Subj!='{}'".format(bs))

fig, ax = plt.subplots(figsize=(38.4,21.6))
sns.barplot(x="Stim", y="SO/deltO_nest", hue="Dur", data=df_sync,
            order=["sham", "fix", "eig"], hue_order=["30s", "2m", "5m"],
            ax=ax)
plt.title("Spindle nesting, SO/deltO ratio\nsynchronised only")
plt.ylim((0,3.5))
plt.savefig("nesting_sync.tif")
