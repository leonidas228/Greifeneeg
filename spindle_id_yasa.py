import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import yasa
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
freq_sp = (12,15)
corr = 0.5
rel_pow = 0.2
rms = 1.5
duration = (0.35,2)

for filename in filelist:
    this_match = re.match("aibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        raw_work = raw.copy()
        ft = raw_work.first_time
        raw_work.pick_channels([channel])
        sp = yasa.spindles_detect(raw_work, freq_sp=freq_sp,
                                  thresh={"corr":corr, "rel_pow":rel_pow,
                                          "rms":rms}, duration=duration,
                                          remove_outliers=True)
        if sp is not None:
            sp = sp.summary()
            for sp_idx in range(len(sp)):
                onset = ft + sp.iloc[sp_idx]["Start"]
                dur= sp.iloc[sp_idx]["Duration"]
                description = "Spindle {}".format(sp_idx)
                raw_work.annotations.append(onset, dur, description)
                onset = ft + sp.iloc[sp_idx]["Peak"]
                description = "Spindle peak {}".format(sp_idx)
                raw_work.annotations.append(onset, 0, description)
                raw.set_annotations(raw_work.annotations)
        raw.save("{}aaibscaf_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond),
                 overwrite=True)
