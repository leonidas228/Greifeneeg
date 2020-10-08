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
conds = ["sham"]
filelist = listdir(proc_dir)

start_times = {"031":615, "033":1020, "035":1209, "021":627, "038":250, "037":618,
               "043":858, "044":1702, "017":840, "045":480, "046":1950, "050":1800,
               "048":2658, "047":798}
stim_time = 150
analy_time = 30
jitter = (240, 360)
stim_nums = 5

for filename in filelist:
    this_match = re.match("f_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        if subj not in start_times.keys():
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        this_start_time = start_times[subj]
        for stim_idx in range(stim_nums):
            raw.annotations.append(this_start_time-analy_time, analy_time,
                                   "Pre_Stimulation {}".format(stim_idx))
            raw.annotations.append(this_start_time, stim_time,
                                   "BAD_Stimulation {}".format(stim_idx))
            raw.annotations.append(this_start_time+stim_time, analy_time,
                                   "Post_Stimulation {}".format(stim_idx))
            this_start_time += stim_time + np.random.randint(*jitter)

        raw.save("{}af_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond),
                 overwrite=True)
