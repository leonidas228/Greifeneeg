import mne
from os import listdir
from os.path import isdir, join
import re
import numpy as np
from gssc.infer import EEGInfer
import matplotlib.pyplot as plt
plt.ion()

"""
Algorithmically determines and marks stimulations on sham data
"""


root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")

stim_duration = 90
analy_duration = 60
gap_idx_range = [1, 3]
min_stims = 5
min_n2 = 4 # minimum minutes of N2 sleep for beginning
min_n2_inter = 1 # same but for interval between stimulations

ei = EEGInfer()
filenames = listdir(proc_dir)
for filename in filenames:
    match = re.match("p_NAP_(\d{4})_(.*)-raw.fif", filename)
    if match:
        (subj, cond) = match.groups()
    else:
        cond = "blah"
    if cond != "sham":
        continue
    raw = mne.io.Raw(join(proc_dir, filename), preload=True)
    raw.filter(l_freq=0.3, h_freq=30)
    stages, times = ei.mne_infer(raw, eeg=["C3", "C4"], eog=["HEOG"])
    hypno_annots = mne.Annotations(times, 30., stages.astype("str"))
    ## mark the sham stimulation and pre/post periods
    stim_annots = mne.Annotations([], [], [])
    # calculate some arrays we'll need
    n2_inds = stages==2
    n2_min_idx = int(np.round(min_n2 / 0.5)) # assumes min_n2 is expressed in minutes; 0.5 is 30s
    n2_min_inter_idx = int(np.round(min_n2_inter / 0.5)) # assumes min_n2_inter is expressed in minutes; 0.5 is 30s
    n2_min = np.array([n2_inds[x-n2_min_idx:x].sum() for x in range(n2_min_idx, len(n2_inds))])
    n2_min_inter = np.array([n2_inds[x-n2_min_inter_idx:x].sum() for x in range(n2_min_inter_idx, len(n2_inds))])
    stim_len_idx = int(np.round((stim_duration+analy_duration)/30))
    # find the first stimulation point
    if not sum(n2_min):
        print(f"Subject {subj} does not appear to sleep. Skipping...")
        continue
    # set current idx to first place with n2_min_idx of consecutive N2 sleep
    cur_idx = np.where(n2_min==n2_min_idx)[0][0] + n2_min_idx # add this because n2_min starts n2_min_idx ahead (see above)
    stim_annots.append(times[cur_idx], stim_duration, "BAD_Stimulation 0")
    stim_annots.append(times[cur_idx]+stim_duration, analy_duration, "Post_Stimulation 0")
    # do subsequent stimuli
    nrem_inds = (stages==2) | (stages==3)
    stim_idx = 1
    last_idx = cur_idx
    cur_idx = last_idx + stim_len_idx + np.random.randint(*gap_idx_range)
    # keep going until 15 stimulations or end of recording
    while stim_idx < 15 and cur_idx < (len(stages)-6):
        # check if Wake happened in the meantime
        if sum(~nrem_inds[last_idx:cur_idx]):
            # there was a wake or REM stage; find the next min_n2_inter stage
            next_n2_idx = np.where(n2_min_inter[cur_idx-n2_min_inter_idx:]==n2_min_inter_idx)[0]
            if not len(next_n2_idx):
                # not another N2; we're done here
                break
            cur_idx = next_n2_idx[0] + cur_idx-n2_min_inter_idx
        stim_annots.append(times[cur_idx], stim_duration, f"BAD_Stimulation {stim_idx}")
        stim_annots.append(times[cur_idx]+stim_duration, analy_duration, 
                           f"Post_Stimulation {stim_idx}")
        last_idx = cur_idx
        cur_idx = last_idx + stim_len_idx + np.random.randint(*gap_idx_range)
        stim_idx += 1
    if stim_idx < min_stims:
        print("\n\nFewer than 5 stimulations could be marked. Not saving...\n\n")
    else:
        print(f"\n\n{stim_idx} stimulations marked\n\n")
        stim_annots.save(join(proc_dir, f"stim_NAP_{subj}_sham-annot.fif"), overwrite=True)

