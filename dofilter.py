import mne
from os import listdir
from os.path import isdir
import re
import numpy as np

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)
proclist = listdir(proc_dir) # and in proc directory
overwrite = False # skip

l_freq = 0.1
h_freq = 200

for filename in filelist:
    this_match = re.search("NAP_(\d{3})_T(\d)-raw.fif",filename)
    if this_match:
        subj, tag = this_match.group(1), this_match.group(2)
        if "f_NAP_{}_T{}-raw.fif".format(subj,tag) in filelist and not overwrite:
            print("Already exists. Skipping.")
            continue
        raw = mne.io.Raw(proc_dir+filename, preload=True)
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs="cuda")
        raw.notch_filter(np.arange(50,h_freq,50), n_jobs="cuda")

        # special cases
        if subj == "021" and tag == "2":
            raw.crop(tmin=0,tmax=5340)

        raw.save("{}f_NAP_{}_T{}-raw.fif".format(proc_dir,subj,tag),
                 overwrite=overwrite)
