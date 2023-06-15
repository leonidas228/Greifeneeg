import mne
from os import listdir
from os.path import join
import re
import numpy as np
from gssc.infer import EEGInfer
import csv

root_dir = "/home/jev/hdd/sfb/"

proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)
proclist = listdir(proc_dir) # and in proc directory
overwrite = False # skip

l_freq = 0.3
h_freq = 30
out_form = "csv"

ei = EEGInfer()
for filename in filelist:
    this_match = re.search("NAP_(\d{3})_T(\d)-raw.fif",filename)
    if this_match:
        subj, tag = this_match.group(1), this_match.group(2)
        fileroot = "NAP_{}_T{}".format(subj,tag)

        raw = mne.io.Raw(proc_dir+filename, preload=True)
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs="cuda")

        # special cases
        if subj == "021" and tag == "2":
            raw.crop(tmin=0,tmax=5340)

        stages, times = ei.mne_infer(raw, eeg=["C3", "C4"], eog=["Li"])

        if out_form == "csv":
            with open(f"{join(proc_dir, fileroot)}.csv", "wt") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([f"Hypnogram of {filename}"])
                csv_writer.writerow(["Epoch", "Time", "Stage"])
                for epo_idx, time, stage in zip(np.arange(len(stages)),
                                                times, stages):
                    csv_writer.writerow([epo_idx, time, stage])
        elif out_form == "mne":
            annot = mne.Annotations(times, 30., stages.astype("str"))
            annot.save(f"{join(proc_dir, fileroot)}-annot.fif")
