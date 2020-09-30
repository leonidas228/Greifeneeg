import mne
import numpy as np
import pickle
from os import listdir
import re
from mne.time_frequency import psd_welch

def sub_channels(raw, picks):
    if len(picks) != 2:
        raise ValueError("Can only take two channels for subtraction")
    data = raw.copy().pick_channels(picks).get_data()
    new_signal = data[0,] - data[1,]
    new_info = mne.create_info(["{}-{}".format(picks[0], picks[1])],
                               raw.info["sfreq"], ch_types="eeg")
    new_raw = mne.io.RawArray(np.expand_dims(new_signal, 0), new_info)
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

root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
conds = ["sham"]
filelist = listdir(proc_dir)

with open("sleep_stage_classifier.pickle", "rb") as f:
    xifer = pickle.load(f)

for filename in filelist:
    this_match = re.match("f_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        sub_channels(raw, ["AFz", "Cz"])
        events = mne.make_fixed_length_events(raw, duration=30)
        epo = mne.Epochs(raw,events).load_data()
        epo.pick_channels(["AFz-Cz"])
        freqs = power_band(epo)
        event_ids = xifer.predict(freqs)
        breakpoint()

        current_stage = None
        begin = 0
        for ev_idx in range(len(events)):
            if event_ids[ev_idx] == current_stage:
                continue
            else:
                current_stage = event_ids[ev_idx]
                end = events[ev_idx,0] - 1
