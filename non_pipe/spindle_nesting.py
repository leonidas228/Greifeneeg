import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()

def annot_subset(annotations, match_str):
    delete_inds = []
    for ann_idx, ann in enumerate(annotations):
        match = re.search(match_str, ann["description"])
        if not match:
            delete_inds.append(ann_idx)
    annotations.delete(delete_inds)
    return annotations

def annot_nearest(annotations, onset):
    nearest = np.inf
    for annot in annotations:
        dist = abs(annot["onset"] - onset)
        if dist > nearest: # things are increasing; stop
            break
        if dist < nearest:
            closest_annot = annot
            nearest = dist
    return closest_annot, nearest

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)

nest_range = (-1,1)

SO_delays = []
SO_nests  =[]
deltO_delays = []
deltO_nests = []
spind_n = 0
spind_SO_n = 0
spind_deltO_n = 0
df_dict = {"Subject":[], "Cond":[], "PrePost":[], "OscType":[], "Index":[], "Delay":[]}
for filename in filelist:
    this_match = re.match("aaibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        spindle_annots = annot_subset(raw.annotations.copy(), "Spindle peak")
        SO_annots = annot_subset(raw.annotations.copy(), "Trough .* SO")
        deltO_annots = annot_subset(raw.annotations.copy(), "Trough .* deltO")
        if not len(SO_annots) or not len(deltO_annots) or not len(spindle_annots):
            continue
        for spind in spindle_annots:
            df_dict["Subject"].append(subj)
            df_dict["Cond"].append(cond)
            spind_n += 1
            onset = spind["onset"]
            nearest_SO, _  = annot_nearest(SO_annots, spind["onset"])
            nearest_deltO, _  = annot_nearest(deltO_annots, spind["onset"])
            if nest_range[0] < (spind["onset"] - nearest_SO["onset"]) < nest_range[1]:
                df_dict["PrePost"].append(re.search("(Post|Pre)",nearest_SO["description"]).groups(0)[0])
                df_dict["Index"].append(int(nearest_SO["description"][-1]))
                df_dict["OscType"].append("SO")
                df_dict["Delay"].append((spind["onset"] - nearest_SO["onset"]))
                SO_delays.append(spind["onset"] - nearest_SO["onset"])
                SO_nests.append(nearest_SO)
                spind_SO_n += 1
            elif nest_range[0] < (spind["onset"] - nearest_deltO["onset"]) < nest_range[1]:
                df_dict["PrePost"].append(re.search("(Post|Pre)",nearest_SO["description"]).groups(0)[0])
                df_dict["Index"].append(int(nearest_SO["description"][-1]))
                df_dict["OscType"].append("deltO")
                df_dict["Delay"].append((spind["onset"] - nearest_deltO["onset"]))
                deltO_delays.append(spind["onset"] - nearest_deltO["onset"])
                deltO_nests.append(nearest_deltO)
                spind_deltO_n += 1
            else:
                df_dict["PrePost"].append("None")
                df_dict["Index"].append(-1)
                df_dict["OscType"].append("None")
                df_dict["Delay"].append(-1)
df = pd.DataFrame.from_dict(df_dict)
df_SO = df.query("OscType=='SO'")
df_deltO = df.query("OscType=='deltO'")
plt.figure()
plt.hist(df_SO["Delay"].values,bins=20,alpha=0.5)
plt.hist(df_deltO["Delay"].values,bins=20,alpha=0.5)
