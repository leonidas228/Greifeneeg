import mne
import numpy as np
from mne.time_frequency import psd_multitaper, tfr_morlet
import matplotlib.pyplot as plt
plt.ion()
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

asyncs = ['015_eig30s', '027_eig30s', '046_eig30s', '033_eig30s', '053_eig30s',
          '025_eig2m', '017_fix2m', '015_fix2m', '044_fix2m', '046_eig2m',
          '022_eig5m', '031_eig5m', '035_eig5m', '046_eig5m', '048_eig5m',
          '044_eig5m', '053_eig5m']

asyncs = ["015_fix5m"]

left_chans = ["FC5", "FC1", "C3", "CP1", "T7", "P3"]
right_chans = ["FC6", "FC2", "C6", "CP2", "T8", "P4"]
n_jobs = 8

for asyn in asyncs:
    infile = "{}lr_bad_caf_NAP_{}-raw.fif".format(proc_dir, asyn)
    try:
        raw = mne.io.Raw(infile)
    except:
        continue

    psds, freqs = psd_multitaper(raw, fmax=2, n_jobs=n_jobs)
    psds = psds.mean(axis=0)
    fmax = freqs[np.argmax(psds)]
    events, descs = mne.events_from_annotations(raw, regexp="BAD_Stimulation")
    if len(events) > 5:
        events = events[:5]

    if "5m" in asyn:
        seconds = 300
    elif "2m" in asyn:
        seconds = 120
    else:
        seconds = 30

    try:
        epo = mne.Epochs(raw, events, baseline=None, tmin=0, tmax=40+seconds,
                         reject_by_annotation=False, preload=True)
    except:
        continue

    epo.plot(scalings={"eeg":3e-3}, butterfly=True)
    raw.plot(scalings={"eeg":3e-3}, butterfly=True)

    left_idx = mne.pick_channels(epo.ch_names, ["left"])
    right_idx = mne.pick_channels(epo.ch_names, ["right"])
    complex = tfr_morlet(epo, [fmax], 3, return_itc=False, n_jobs=4,
                         average=False, output="complex")
    phase = np.angle(complex.data)
    phase_diff = np.angle(np.exp(0+1j*(phase[:, left_idx,] - phase[:, right_idx,])))[:,0,]
    phase_diff = phase_diff.squeeze()
    print("hi")
    plt.figure()
    plt.plot(phase_diff.T)

    breakpoint()
    plt.close("all")
