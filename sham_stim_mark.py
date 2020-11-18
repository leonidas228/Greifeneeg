import mne
from os import listdir
import re
import numpy as np
from os.path import isdir
import pandas as pd

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["sham"]
filelist = listdir(proc_dir)

start_times = {"002":360, "003":5640, "005":930, "006":1780, "007":540, "009":1110,
               "013":960, "016":720, "015":500, "018":1560, "022":570, "024":1290,
               "025":675, "026":2400, "028":729, "031":615, "033":1020, "035":1209,
               "021":627, "038":250, "037":618, "043":858, "044":1702, "017":840,
               "045":480, "046":1950, "050":1800, "048":2658, "047":798}

df = pd.read_pickle("{}start_times.pickle".format(proc_dir))

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
        start_vals = df.query("Subj=='{}'".format(subj))["Start"].values
        med = np.percentile(start_vals, 0.5)
        start_vals = start_vals[(med-180 < start_vals) & (start_vals < med+180)]
        this_start_time = start_vals.mean()
        if not len(start_vals):
            raise ValueError("No median could be found.")
        raw = mne.io.Raw(proc_dir+filename,preload=True)
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
