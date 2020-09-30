import mne
from os import listdir
from os.path import isdir
import re

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
raw_dir = root_dir+"raw/"
proc_dir = root_dir+"proc/"
filelist = listdir(raw_dir)
proclist = listdir(proc_dir)


for filename in filelist:
    this_match = re.search("NAP_(\d{3})_T(\d)(b?).vhdr",filename)
    if this_match:
        subj, tag = this_match.group(1), this_match.group(2)
        if "NAP_{}_T{}-raw.fif".format(subj,tag) in proclist:
            print("Already exists. Skipping.")
            continue
        raw = mne.io.read_raw_brainvision(raw_dir+filename)
        raw.save("{}NAP_{}_T{}-raw.fif".format(proc_dir,subj,tag),overwrite=True)
