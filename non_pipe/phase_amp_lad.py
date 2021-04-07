import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
from scipy.signal import hilbert
from mne.time_frequency import psd_multitaper, tfr_morlet
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
minmax_freqs = {"SO":(0.5, 1.25), "deltO":(0.75, 4.25)}
chan = "frontal"
osc = "SO"
epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)

psds, freqs = psd_multitaper(epo, tmin=-.5, fmin=minmax_freqs[osc][0],
                             fmax=minmax_freqs[osc][1], picks=[chan],
                             n_jobs=n_jobs, adaptive=True)
fmaxs = []
for epo_idx in range(psds.shape[0]):
    fmaxs.append(freqs[np.argmax(psds[epo_idx,0])])
fmaxs = np.array(fmaxs)

tfr_eeg = tfr_morlet(epo, [fmaxs.mean()], 1, n_jobs=n_jobs,
                     picks=chan, return_itc=False, output="complex",
                     average=False)
data = tfr_eeg.data[:,0,0,]
power = (data * data.conj()).real
phase = np.angle(data)
