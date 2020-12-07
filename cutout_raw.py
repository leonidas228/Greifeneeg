import mne
from os import listdir
import re
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s", "sham"]
conds = ["sham30s", "sham2m", "sham5m"]
filelist = listdir(proc_dir)
excludes = ["031_eig30s", "045_fix5m", "046_eig30s"]
excludes = []
overwrite = True

for filename in filelist:
    this_match = re.match("af_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds or "{}_{}".format(subj,cond) in excludes:
            continue
        if "caf_NAP_{}_{}-raw.fif".format(subj,cond) in filelist and not overwrite:
            print("Already exists. Skipping.")
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        raws = []
        for annot in raw.annotations:
            match = re.match("(.*)_Stimulation (\d)", annot["description"])
            if match:
                stim_pos, stim_idx = match.group(1), match.group(2)
                if stim_pos == "BAD":
                    continue
                begin, duration = annot["onset"], annot["duration"]
                end = begin + duration
                if end > raw.times[-1]:
                    end = raw.times[-1]
                raws.append(raw.copy().crop(begin,end))
        if len(raws) == 0:
            continue
        raw_cut = raws[0]
        raw_cut.append(raws[1:])
        raw_cut.save("{}caf_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond),
                     overwrite=overwrite)
