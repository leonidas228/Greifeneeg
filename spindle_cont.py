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
thresh = 99.9
epo_pref = "ak_"
epo_pref = ""

for chan in chans:
    epo = mne.read_epochs("{}{}grand_{}-epo.fif".format(proc_dir, epo_pref,chan),
                          preload=True)
    epo.resample(sfreq, n_jobs="cuda")
    power = tfr_morlet(epo, spindle_freq, n_cycles=5, average=False,
                       return_itc=False, n_jobs=n_jobs)
    power.crop(tmin=-2.15, tmax=1.65)
    power.apply_baseline((-2.15,-1.68), mode="logratio")
    power.crop(tmin=-1.5,tmax=1.5)
    power.save("{}{}grand_{}-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    vals = power.data[:,0,].mean(axis=1).flatten()
    plt.hist(vals, bins=100)

    pos_thresh = np.percentile(vals, thresh)
    neg_thresh = np.percentile(vals, 100-thresh)
