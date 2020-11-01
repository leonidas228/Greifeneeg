import mne
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 4
spindle_freq = np.arange(10,20)
chans = ["frontal", "central"]
osc_types = ["SO", "deltO"]

for chan in chans:
    epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan),
                          preload=True)
    power = tfr_morlet(epo, spindle_freq, n_cycles=5, average=False,
                       return_itc=False, n_jobs=n_jobs)
    power.apply_baseline((-1.5,-1.),mode="zscore")
    power.save("{}grand_{}-tfr.h5".format(proc_dir, chan), overwrite=True)
