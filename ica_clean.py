import mne
from os import listdir
import re
import numpy as np
from mne.preprocessing import read_ica
import matplotlib.pyplot as plt
plt.ion()
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
conds = ["sham"]
filelist = listdir(proc_dir)

input_dict = {}
for filename in filelist:
    this_match = re.search("bscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        rawfile = "bscaf_NAP_{}_{}-raw.fif".format(subj,cond)
        icafile = "bscaf_NAP_{}_{}-ica.fif".format(subj,cond)
        if (rawfile not in filelist or icafile not in filelist) and show_ica:
            raise ValueError("Not a full pair for subject {}".format(subj))
        raw = mne.io.Raw(proc_dir+rawfile, preload=True)
        ica = read_ica(proc_dir+icafile)
        bad_inds, scores = ica.find_bads_eog(raw)
        new_raw = ica.apply(raw, exclude=bad_inds)
        new_raw.save("{}ibscaf_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond),overwrite=True)
