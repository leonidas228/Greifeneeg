import mne
from os import listdir, getcwd
from os.path import isdir, join
import re
import numpy as np
import matplotlib.pyplot as plt
from utils import get_ptp_annotations

"""
Filter, resample, and organise the channels
"""
root_dir = getcwd()
proc_dir = join(root_dir, "data/proc")

l_freq = 0.1
h_freq = 100
n_jobs = "cuda" # change this to 1 or some higher integer if you don't have CUDA
sfreq = 200

overwrite = True
filelist = listdir(proc_dir)
sfreqs = {}
for filename in filelist:
    this_match = re.match("NAP_(\d{4})_(.*)-raw.fif", filename)
    if not this_match:
        continue
    (subj, cond) = this_match.groups()
    outfile = f"p_NAP_{subj}_{cond}-raw.fif"
    if outfile in filelist and not overwrite:
        print("Already exists. Skipping.")
        continue


    # filtering and resampling
    raw = mne.io.Raw(join(proc_dir, filename), preload=True)
    raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
    raw.notch_filter(np.arange(50, h_freq+50 ,50), n_jobs=n_jobs)
    sfreqs[filename] = raw.info["sfreq"]
    raw.resample(sfreq, n_jobs=n_jobs)
    


    # create EOG/EMG chanenls
    if "HEOG" not in raw.ch_names:
        raw = mne.set_bipolar_reference(raw, "Li", "Re", ch_name="HEOG")
        raw.set_channel_types({"HEOG":"eog"})
    if "Mov" not in raw.ch_names:
        raw = mne.set_bipolar_reference(raw, "MovLi", "MovRe", ch_name="Mov")
        raw.set_channel_types({"Mov":"emg"})
    if "VEOG" not in raw.ch_names and "Vo" in raw.ch_names and "Vu" in raw.ch_names:
        raw = mne.set_bipolar_reference(raw, "Vo", "Vu", ch_name="VEOG")
        raw.set_channel_types({"VEOG":"eog"})
    raw.save(join(proc_dir, outfile), overwrite=overwrite)
    
    bad_jumps= mne.preprocessing.annotate_amplitude(raw, peak=5e-5, min_duration=0.005)
    bad_channels = bad_jumps[1]
    bad_jumps=bad_jumps[0]
    
    #bad_segs = get_ptp_annotations(raw, ptp_thresh=5e-4, duration = 5)
    bad_mov = get_ptp_annotations(raw, ptp_thresh=15e-5, duration = 1, channels = "Mov")
    #raw.set_annotations(bad_segs)
    #raw.set_annotations(bad_mov)   
    #bad= merge_annotations(bad_jumps, bad_mov)
    print(bad_mov.orig_time) 
    print('+++++++++++++++++++++++++')
    print(bad_jumps.orig_time)
    bad= bad_jumps.__add__(bad_mov)
    raw.set_annotations(bad)
    raw.plot(block=True)

    break 
