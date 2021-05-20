import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_welch
from os.path import isdir
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

def append_dicts(dict_a, dict_b):
    for k,v in dict_a["freqs"].items():
        for kk,vv in v.items():
            dict_a["freqs"][k][kk] = np.concatenate((dict_a["freqs"][k][kk],
                                                     dict_b["freqs"][k][kk]))
    for k,v in dict_a["amps"].items():
        dict_a["amps"][k] = np.concatenate((dict_a["amps"][k],
                                            dict_b["amps"][k]))
    return dict_a

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

    eeg_picks = mne.pick_types(epo.info, eeg=True)
    eeg_ch_names = [epo.ch_names[idx] for idx in np.nditer(eeg_picks)]
    psds, freqs = psd_welch(epochs, picks=eeg_picks, fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = {ch_name:{band_name:[] for band_name in FREQ_BANDS.keys()}
         for ch_name in eeg_ch_names}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        freq_inds = (freqs >= fmin) & (freqs < fmax)
        for ch_idx, ch_name in zip(eeg_picks, eeg_ch_names):
            psds_band = psds[:, ch_idx, freq_inds].mean(axis=-1)
            X[ch_name][band_name] = psds_band.reshape(len(psds), -1)
    return X

def epo_amps(epo):
    chans = ["EOG horizontal", "EMG submental"]
    picks = mne.pick_channels(epo.ch_names, chans)
    X = {}
    data = epo.load_data().get_data()
    for ch, pick in zip(chans, np.nditer(picks)):
        X[ch] = np.sqrt(np.sum(data[:,pick,]**2, axis=-1))
    return X

subjects = list(np.arange(83))
subjects = [subj for subj in subjects if subj not in [36,39,52,68,69,78,79]]

desc_2_event_id = {'Sleep stage W': 1,
                   'Sleep stage 1': 2,
                   'Sleep stage 2': 3,
                   'Sleep stage 3': 4,
                   'Sleep stage 4': 4,
                   'Sleep stage R': 5}

event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

all_files = fetch_data(subjects=subjects, recording=[1])
if isdir("/home/jev"):
    proc_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    proc_dir = "/home/jeff/hdd/jeff/sfb/"

ys = []
for file_idx, files in enumerate(all_files):
    mapping = {'EOG horizontal': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}

    raw = mne.io.read_raw_edf(files[0])
    #raw.pick_channels(["EEG Fpz-Cz"])
    annot = mne.read_annotations(files[1])

    raw.set_annotations(annot, emit_warning=False)
    raw.set_channel_types(mapping)
    events, _ = mne.events_from_annotations(raw, event_id=desc_2_event_id,
                                            chunk_duration=30.)
    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    try:
        epo = mne.Epochs(raw=raw, events=events, event_id=event_id,
                         tmin=0., tmax=tmax, baseline=None)
    except:
        continue
    freqs = power_band(epo)
    amps = epo_amps(epo)
    this_mat = {"freqs":freqs, "amps":amps}

    if file_idx:
        mats = append_dicts(mats, this_mat)
    else:
        mats = this_mat
    ys.append(events[:,2])

ys = np.hstack(ys)
with open("{}freq_mat.pickle".format(proc_dir), "wb") as f:
    pickle.dump(mats, f)
np.save("{}ys.npy".format(proc_dir), ys)
