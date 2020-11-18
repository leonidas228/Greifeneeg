import mne
from os import listdir
from os.path import isdir
import re
import numpy as np

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"

raw_dir = root_dir+"raw/30s_sham/" # get raw files from here
proc_dir = root_dir+"proc/" # save the processed files here
filelist = listdir(raw_dir) # get list of all files in raw directory
proclist = listdir(proc_dir) # and in proc directory
overwrite = True # skip
analy_time = 30
dur_key = {"30s":30, "2m":120, "5m":300}
l_freq = 0.1
h_freq = 200

for filename in filelist: # cycle through all files in raw directory
    this_match = re.search("NAP_(\d{3})_T(\d)(b|c?).*.vhdr", filename)
    # do something if the file fits the raw file pattern
    if this_match:
        # pull subject and tag out of the filename and assign to variables
        subj, tag = this_match.group(1), this_match.group(2)
        if "NAP_{}_T{}-raw.fif".format(subj,tag) in proclist and not overwrite:
            print("Already exists. Skipping.")
            continue
        raw = mne.io.read_raw_brainvision(raw_dir+filename, preload=True) # convert
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs="cuda")
        raw.notch_filter(np.arange(50,h_freq,50), n_jobs="cuda")
        these_annotations = raw.annotations.copy()
        stim_count = 0
        for annot in raw.annotations:
            if "Start" in annot["description"]:
                onset = annot["onset"]
                annot_match = re.search("(\d*s|m)_Stim(\d).*", annot["description"])
                duration = dur_key[annot_match.group(1)]
                stim_idx = int(annot_match.group(2))
                these_annotations.append(onset, duration,
                                         "BAD_Stimulation {}".format(stim_idx))
                these_annotations.append(onset - analy_time, analy_time,
                                         "Pre_Stimulation {}".format(stim_idx))
                these_annotations.append(onset + duration, analy_time,
                                         "Post_Stimulation {}".format(stim_idx))
                stim_count += 1
        raw.set_annotations(these_annotations)
        print("\n\nMarked {} stimulations\n\n".format(stim_count))
        if stim_count < 5:
            breakpoint()
        raw.save("{}af_NAP_{}_sham-raw.fif".format(proc_dir, subj, tag),
                 overwrite=overwrite) # save
