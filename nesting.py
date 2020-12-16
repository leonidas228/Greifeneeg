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

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)

channel = "central"
SO_border = [-1, 1]
deltO_border = [-.5,.5]

df_dict = {"Subj":[], "Cond":[], "SO_nest":[], "deltO_nest":[], "SO/deltO_nest":[]}
for filename in filelist:
    this_match = re.match("NAP_(\d{3})_(.*)_(.*)_(.*)-raw.fif",filename)
    if this_match:
        subj, cond, ort, this_osc = this_match.group(1), this_match.group(2), this_match.group(3), this_match.group(4)
        if ort != channel or this_osc != "SO":
            continue
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
        df_dict["SO_nest"].append(SO_ratio)
        df_dict["deltO_nest"].append(deltO_ratio)
        df_dict["SO/deltO_nest"].append(SO_ratio/deltO_ratio)

df = pd.DataFrame.from_dict(df_dict)
