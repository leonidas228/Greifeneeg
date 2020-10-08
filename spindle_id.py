import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)

l_freq = 0.3
h_freq = None
channel = "central"
epolen = 60
n_jobs = 8
spindle_freqs = np.arange(10,15)

for filename in filelist:
    this_match = re.match("aibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        raw.pick_channels([channel])
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
        epo = mne.make_fixed_length_epochs(raw, duration=epolen)
        power = tfr_morlet(epo, spindle_freqs, n_cycles=5, average=False,
                           return_itc=False, n_jobs=n_jobs)

        tfr = np.zeros(0)
        for epo_tfr in power.__iter__():
            tfr = np.concatenate((tfr,np.mean(epo_tfr[:,0,],axis=0)))
        tfr_aschan = np.zeros(len(raw))
        tfr_aschan[:len(tfr)] = tfr
