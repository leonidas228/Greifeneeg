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
raw_dir_txt = proc_dir+"sleep_stages/"
conds = ["sham"]
filelist = listdir(proc_dir)
rawtxtlist = listdir(raw_dir_txt)

stim_time = 5
analy_time = 1
interval = 3
stim_num = 5
first_buffer = 8
second_buffer = 2

for rawtxt in rawtxtlist: # cycle through all files in raw directory
    this_match = re.search("NAP_(\d{3}).*txt", rawtxt)
    if this_match:
        with open(raw_dir_txt+rawtxt, "rt") as f:
            labels = f.readlines()
            labels = [line.rstrip("\n") for line in labels]
            labels = np.array([(int(line[0]),int(line[-1])) for line in labels])
            no_movs = labels[:,1] == 0
            right_stage = (labels[:,0] >= 2) & (labels[:,0] <= 4)
        subj_txt = this_match.group(1)
        # do something if the file fits the raw file pattern
        for filename in filelist:
            this_match = re.match("f_NAP_(\d{3})_(.*)-raw.fif",filename)
            if this_match:
                subj, cond = this_match.group(1), this_match.group(2)
                if subj == subj_txt and cond=="sham":
                    raw = mne.io.Raw(proc_dir+filename,preload=True)
                    stim_idx = 0
                    schnitt_idx = 0
                    while (stim_idx < stim_num) and (schnitt_idx < len(labels)):
                        if stim_idx:
                            buffer = second_buffer
                        else:
                            buffer = first_buffer
                        stim_len = buffer + analy_time*2 + stim_time
                        if (schnitt_idx + stim_len) >= len(right_stage):
                            break
                        if all(right_stage[schnitt_idx:schnitt_idx+stim_len]):
                            #breakpoint()
                            raw.annotations.append((buffer+schnitt_idx)*30, analy_time*30,
                                                   "Pre_Stimulation {}".format(stim_idx))
                            raw.annotations.append((buffer+schnitt_idx+analy_time)*30,
                                                   stim_time*30,
                                                   "BAD_Stimulation {}".format(stim_idx))
                            raw.annotations.append((buffer+schnitt_idx+analy_time+stim_time)*30,
                                                   analy_time*30,
                                                   "Post_Stimulation {}".format(stim_idx))
                            schnitt_idx += stim_len + interval
                            stim_idx += 1
                        else:
                            schnitt_idx += 1
                    print("{}af_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond))
                    if subj == "028":
                        breakpoint()
                    raw.save("{}af_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond),
                             overwrite=True)
