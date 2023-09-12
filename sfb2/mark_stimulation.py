import mne
from os import listdir
from os.path import isdir, join
import re
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()

"""
Marks stimulation and post-stimulation periods on the basis of the triggers
"""

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")
analy_duration = 60
between_duration = 60
filelist = listdir(proc_dir)
post_only = True
exclude = ["1038", "1026", "1036"]
exclude = []
overwrite = True

pre_trig = 70 # go back this far from the trigger
duration = 100

filenames = listdir(proc_dir)
for filename in filenames:
    match = re.match("p_NAP_(\d{4})_(.*)-raw.fif", filename)
    if not match:
        continue
    (subj, cond) = match.groups()
    if cond == "sham" or subj in exclude:
        continue

    outname = f"stim_NAP_{subj}_{cond}-annot.fif"
    if outname in filenames and not overwrite:
        print(f"{outname} already exists. Skipping...")
        continue
    raw = mne.io.Raw(join(proc_dir, filename), preload=True)
    stim_annots = mne.Annotations([], [], [])
    stim_idx = 0
    for annot in raw.annotations:
        if "Response/R128" in annot["description"]:
            stim_annots.append(annot["onset"]-pre_trig, duration, f"BAD_Stimulation {stim_idx}")
            stim_annots.append(annot["onset"]-pre_trig+duration, analy_duration, 
                               f"Post_Stimulation {stim_idx}")
            stim_idx += 1

    stim_annots.save(join(proc_dir, outname), overwrite=overwrite)