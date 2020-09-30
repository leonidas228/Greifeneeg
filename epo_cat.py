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

epos = []
for filename in filelist:
    this_match = re.match("d_NAP_(\d{3})_(.*)-epo.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        epo = mne.read_epochs(proc_dir+filename)
        epo.pick_channels([chan])
        epos.append(epo)
grand_epo = mne.concatenate_epochs(epos)
grand_epo.save("{}grand-epo.fif".format(proc_dir), overwrite=True)
