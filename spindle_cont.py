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

n_jobs = 8
spindle_freq = np.arange(10,20)
chans = ["central"]
osc_types = ["SO", "deltO"]
sfreq = 50.

for chan in chans:
    epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan),
                          preload=True)
    epo.resample(sfreq, n_jobs="cuda")
    power = tfr_morlet(epo, spindle_freq, n_cycles=5, average=False,
                       return_itc=False, n_jobs=n_jobs)
    power.crop(tmin=-1.35,tmax=1.25)
    power.apply_baseline((-1.35,-1))
    power.crop(tmin=-1,tmax=1)
    power.save("{}grand_{}-tfr.h5".format(proc_dir, chan), overwrite=True)
