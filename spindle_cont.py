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
spindle_freq = np.arange(10,21)
chans = ["central"]
osc_types = ["SO", "deltO"]
sfreq = 50.
thresh = 99.9
epo_pref = ""

bl = (-2.15,-1.2)
crop = (-1.2,1.2)

for chan in chans:
    epo = mne.read_epochs("{}{}grand_{}_finfo-epo.fif".format(proc_dir, epo_pref,chan),
                          preload=True)
    epo.resample(sfreq, n_jobs="cuda")
    power = tfr_morlet(epo, spindle_freq, n_cycles=5, average=False,
                       return_itc=False, n_jobs=n_jobs)
    power.crop(tmin=-2.15, tmax=1.2) # get rid of edge effects

    power_mean = power.copy().apply_baseline((bl[0],bl[1]), mode="mean")
    power_mean.crop(tmin=crop[0],tmax=crop[1])
    power_mean.save("{}{}grand_{}_mean-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_log = power.copy()
    power_log.data = np.log10(power_log.data)
    power_log = power_log.apply_baseline((bl[0],bl[1]), mode="mean")
    power_log.crop(tmin=crop[0],tmax=crop[1])
    power_log.save("{}{}grand_{}_logmean-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_z = power.copy().apply_baseline((bl[0],bl[1]), mode="zscore")
    power_z.crop(tmin=crop[0],tmax=crop[1])
    power_z.save("{}{}grand_{}_zscore-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_logratio = power.copy().apply_baseline((bl[0],bl[1]), mode="logratio")
    power_logratio.crop(tmin=crop[0],tmax=crop[1])
    power_logratio.save("{}{}grand_{}_logratio-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_zlogratio = power.copy().apply_baseline((bl[0],bl[1]), mode="zlogratio")
    power_zlogratio.crop(tmin=crop[0],tmax=crop[1])
    power_zlogratio.save("{}{}grand_{}_zlogratio-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power.crop(tmin=crop[0],tmax=crop[1])
    power.save("{}{}grand_{}_none-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)
