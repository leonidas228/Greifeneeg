import mne
import numpy as np
import pickle
from os import listdir
from os.path import isdir
import re
from mne.time_frequency import psd_welch

def sub_channels(raw, picks, name, ch_type="eeg"):
    if len(picks) != 2:
        raise ValueError("Can only take two channels for subtraction")
    to_sub = []
    for pick in picks:
        if type(pick) is list:
            sub0 = raw.copy().pick_channels(pick).get_data()
            to_sub.append(np.mean(sub0, axis=0))
        else:
            to_sub.append(raw.copy().pick_channels([pick]).get_data())
    new_signal = to_sub[0] - to_sub[1]
    new_info = mne.create_info([name], raw.info["sfreq"], ch_types=ch_type)
    new_raw = mne.io.RawArray(new_signal, new_info)
    raw.add_channels([new_raw], force_update_info=True)


def power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

def epo_amps(epo):
    picks = mne.pick_channels(epo.ch_names, ["EOG horizontal", "EMG submental"])
    data = epo.load_data().get_data()[:,picks,]
    return np.sqrt(np.sum(data**2, axis=-1))

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"/proc/"
conds = ["sham"]
subjs = ["018", "026", "048"]
filelist = listdir(proc_dir)

with open("sleep_stage_classifier.pickle", "rb") as f:
    xifer = pickle.load(f)

for filename in filelist:
    this_match = re.match("af_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        if subj not in subjs:
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        sub_channels(raw, [["Fp1", "Fp2"], "Cz"], "Fpz - Cz")
        sub_channels(raw, ["Pz", ["O2", "O1"]], "Pz - Oz")
        sub_channels(raw, ["Li", "Re"], "EOG horizontal", ch_type="eog")
        sub_channels(raw, ["MovLi", "MovRe"], "EMG submental", ch_type="misc")
        raw.pick_channels(["Fpz - Cz", "Pz - Oz", "EOG horizontal", "EMG submental"])
        events = mne.make_fixed_length_events(raw, duration=30)
        epo = mne.Epochs(raw,events).load_data()
        events = epo.events
        freqs = power_band(epo)
        amps = epo_amps(epo)
        mat = np.hstack((freqs,amps))
        event_ids = xifer.predict(mat)

        current_stage = None
        begin = 0
        for ev_idx in range(len(events)):
            if event_ids[ev_idx] == current_stage:
                continue
            else:
                current_stage = event_ids[ev_idx]
                end = events[ev_idx,0] - 1
        events[:,-1] = event_ids
        breakpoint()
