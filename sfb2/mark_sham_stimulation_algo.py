import mne
from os import listdir
from os.path import isdir, join
import re
import numpy as np
from gssc.infer import EEGInfer
import matplotlib.pyplot as plt
plt.ion()

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")

stim_duration = 120
analy_duration = 60
gap_idx_range = [2, 5]
min_stims = 5

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
    stages, times = ei.mne_infer(raw, eeg=["C3", "C4"], eog=["Li"])
    hypno_annots = mne.Annotations(times, 30., stages.astype("str"))
    ## mark the sham stimulation and pre/post periods
    stim_annots = mne.Annotations([], [], [])
    # find the first stimulation point
    n2_inds = stages==2
    n2_2m = np.array([n2_inds[x-4:x].sum() for x in range(4, len(n2_inds))])
    if not sum(n2_2m):
        print(f"Subject {subj} does not appear to sleep. Skipping...")
        continue
    cur_idx = np.where(n2_2m==4)[0][0] + 4
    stim_annots.append(times[cur_idx], stim_duration, "BAD_Stimulation 0")
    stim_annots.append(times[cur_idx]+stim_duration, analy_duration, "Post_Stimulation 0")
    # do subsequent stimuli
    nrem_inds = (stages==2) | (stages==3)
    stim_idx = 1
    last_idx = cur_idx
    cur_idx = last_idx + 4 + np.random.randint(*gap_idx_range)
    # keep going until 15 stimulations or end of recording
    while stim_idx <= 15 and cur_idx < (len(stages)-6):
        # check if Wake happened in the meantime
        if sum(~nrem_inds[last_idx:cur_idx]):
            # there was a wake or REM stage; find the next N2 stage
            next_n2_idx = np.where(n2_inds[cur_idx:])[0]
            if not len(next_n2_idx):
                # not another N2; we're done here
                break
            cur_idx = next_n2_idx[0] + cur_idx
        stim_annots.append(times[cur_idx], stim_duration, f"BAD_Stimulation {stim_idx}")
        stim_annots.append(times[cur_idx]+stim_duration, analy_duration, 
                           f"Post_Stimulation {stim_idx}")
        last_idx = cur_idx
        cur_idx = last_idx + 4 + np.random.randint(*gap_idx_range)
        stim_idx += 1
    if stim_idx < min_stims:
        print("\n\nFewer than 5 stimulations could be marked. Not saving...\n\n")
    else:
        print(f"\n\n{stim_idx} stimulations marked\n\n")
        stim_annots.save(join(proc_dir, f"stim_NAP_{subj}_sham-annot.fif"), overwrite=True)

