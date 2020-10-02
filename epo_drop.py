import mne
from os import listdir
import re
import numpy as np
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
filelist = listdir(proc_dir)
chan = "central"

for filename in filelist:
    this_match = re.match("NAP_(\d{3})_(.*)-epo.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        epo = mne.read_epochs(proc_dir+filename)
        pick = mne.pick_channels(epo.ch_names,[chan])[0]
        epo_data = epo.get_data()[:,pick,]
        epo_data_flat = np.abs(epo_data.flatten())
        thresh = np.percentile(epo_data_flat, 99.9)
        thresh = 1e-4
        print("Threshold: {}".format(thresh))
        drop_inds = []
        for epo_idx in range(len(epo)):
            if np.abs(epo_data[epo_idx,]).max() > thresh:
                drop_inds.append(epo_idx)
        print(drop_inds)
        epo.drop(drop_inds)
        epo.save("{}d_NAP_{}_{}-epo.fif".format(proc_dir,subj,cond), overwrite=True)
