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


proc_dir = root_dir+"proc/" # save the processed files here
proclist = listdir(proc_dir) # and in proc directory
overwrite = True # skip
analy_time = 60
dur_key = {"30s":30, "2m":120, "5m":300}
l_freq = 0.1
h_freq = 200
post_only = True

bads = []
for dur in dur_key.keys():
    raw_dir = root_dir+"raw/{}_sham/".format(dur) # get raw files from here
    filelist = listdir(raw_dir) # get list of all files in raw directory
    duration = dur_key[dur]
    for filename in filelist: # cycle through all files in raw directory
        this_match = re.search("NAP_(\d{3})_T(\d)(b|c?).*.vhdr", filename)
        # do something if the file fits the raw file pattern
        if this_match:
            # pull subject and tag out of the filename and assign to variables
            subj, tag = this_match.group(1), this_match.group(2)
            if "af_NAP_{}_sham{}-raw.fif".format(subj,dur) in proclist and not overwrite:
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
                    annot_match = re.search("Stim(\d).*", annot["description"])
                    if not annot_match:
                        print("\n\nIncorrect trigger for subject {}, {}\n\n".format(subj, dur))
                        bads.append(filename)
                        continue
                    stim_idx = int(annot_match.group(1))-1
                    if post_only:
                        these_annotations.append(onset, duration,
                                                 "BAD_Stimulation {}".format(stim_idx))
                        if stim_idx == 0:
                            these_annotations.append(onset - analy_time*4, analy_time*4,
                                                     "Pre_Stimulation {}".format(stim_idx))
                        these_annotations.append(onset + duration, analy_time,
                                                 "Post_Stimulation {}".format(stim_idx))
                    else:
                        these_annotations.append(onset, duration,
                                                 "BAD_Stimulation {}".format(stim_idx))
                        these_annotations.append(onset - analy_time, analy_time,
                                                 "Pre_Stimulation {}".format(stim_idx))
                        these_annotations.append(onset + duration, analy_time,
                                                 "Post_Stimulation {}".format(stim_idx))
                    stim_count += 1
            raw.set_annotations(these_annotations)
            print("\n\nMarked {} stimulations\n\n".format(stim_count))
            raw.save("{}af_NAP_{}_sham{}-raw.fif".format(proc_dir, subj, dur),
                     overwrite=overwrite) # save
