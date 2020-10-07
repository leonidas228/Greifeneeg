import mne
from mne.preprocessing import ICA
from os import listdir
import re
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
conds = ["sham"]
filelist = listdir(proc_dir)
n_jobs = 8

for filename in filelist:
    this_match = re.match("bscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        ica = ICA(method="picard",max_iter=500)
        ica.fit(raw)
        ica.save("{}bscaf_NAP_{}_{}-ica.fif".format(proc_dir,subj,cond))
