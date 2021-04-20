import pandas as pd
import mne
import matplotlib.pyplot as plt
plt.ion()
from os import listdir
from os.path import isdir
from anoar import BadChannelFind
import re
import numpy as np
from scipy.signal import find_peaks

class OscEvent():
    def __init__(self, start_time, end_time, peak_time, peak_amp, trough_time,
                 trough_amp):
        self.start_time = start_time
        self.end_time = end_time
        self.peak_time = peak_time
        self.peak_amp = peak_amp
        self.trough_time = trough_time
        self.trough_amp = trough_amp
        self.event_id = None
        self.event_annot = None

def check_trough_annot(desc):
    event = None
    if "Trough" in desc:
        event_idx = int(desc[-1])
        event = event_idx * 200
        if "Post" in desc:
            event += 150
        event += event_idx
    return event

def get_annotation(annotations, time):
    period = None
    for annot in annotations:
        if "Pre" not in annot["description"] and "Post" not in annot["description"]:
            continue
        begin = annot["onset"]
        end = begin + annot["duration"]
        if time > begin and time < end:
            period = annot["description"]
    return period

def osc_peaktroughs(osc_events):
    peaks = []
    troughs = []
    for oe in osc_events:
        peaks.append(oe.peak_amp)
        troughs.append(oe.trough_amp)
    peaks, troughs = np.array(peaks), np.array(troughs)
    return peaks, troughs

def mark_osc_amp(osc_events, amp_thresh, chan_name, minmax_times, osc_type):
    osc_idx = 0
    for oe in osc_events:
        pt_time_diff = oe.trough_time - oe.peak_time
        time_diff = oe.end_time - oe.start_time
        pt_amp_diff = oe.peak_amp - oe.trough_amp
        if pt_amp_diff > amp_thresh and minmax_time[0] < time_diff < minmax_time[1]:
            oe.event_id = "{} {} {}".format(chan_name, osc_type, osc_idx)
            osc_idx += 1

def find_densest(annots, winlen, incr=10):
    max_count = 0
    start_stim = 0
    onsets = np.array([annot["onset"] for annot in new_annots])
    for start_time in range(0, int(onsets[-1]), incr):
        count = len(np.where((onsets > start_time) & (onsets < start_time+winlen))[0])
        if count > max_count:
            start_stim = start_time
            max_count = count
    print()
    return start_stim


if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

df = pd.read_pickle("{}amp_thresh.pickle".format(proc_dir))
filelist = listdir(proc_dir)
chan_groups = {"frontal":["Fz","FC1","FC2"],
               "central":["Cz","CP1","CP2"]}
min_samples = 30
minmax_freqs = [(0.16, 1.25), (0.75, 4.25)]
minmax_times = [(1, 2), (0.25, 1)]
osc_types = ["SO", "deltO"]
start_dict = {"Subj":[],"OscType":[],"Chan":[],"Start":[]}
overwrite = False

for filename in filelist:
    this_match = re.match("f_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond != "sham":
            continue
        if "caf_NAP_{}_{}-raw.fif".format(subj,cond) in filelist and not overwrite:
            print("Already exists. Skipping.")
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        picks = mne.pick_types(raw.info, eeg=True)
        bcf = BadChannelFind(picks, thresh=0.5)
        bad_chans = bcf.recommend(raw)
        print(bad_chans)
        raw.info["bads"].extend(bad_chans)
        for k,v in chan_groups.items():
            raw_pick = raw.copy().pick_channels(v)
            avg_signal = raw_pick.get_data().mean(axis=0, keepdims=True)
            avg_info = mne.create_info([k], raw.info["sfreq"], ch_types="eeg")
            avg_raw = mne.io.RawArray(avg_signal, avg_info)
            raw.add_channels([avg_raw], force_update_info=True)
        for minmax_freq, minmax_time, osc_type in zip(minmax_freqs,
                                                      minmax_times,
                                                      osc_types):
            raw_work = raw.copy()
            raw_work.filter(l_freq=minmax_freq[0], h_freq=minmax_freq[1], n_jobs=8)
            first_time = raw_work.first_samp / raw_work.info["sfreq"]

            # zero crossings
            for k in chan_groups.keys():
                pick_ind = mne.pick_channels(raw_work.ch_names, include=[k])
                signal = raw_work.get_data()[pick_ind,].squeeze()

                # need to add infinitesimals to zeros to prevent weird x-crossing bugs
                for null_idx in list(np.where(signal==0)[0]):
                    if null_idx:
                        signal[null_idx] = 1e-16*np.sign(signal[null_idx-1])
                    else:
                        signal[null_idx] = 1e-16*np.sign(signal[null_idx+1])

                zero_x_inds = (np.where((signal[:-1] * signal[1:]) < 0)[0]) + 1
                # cycle through negative crossings
                neg_x0_ind = 1 if signal[0] < 0 else 2
                osc_events = []
                for zx_ind in range(neg_x0_ind, len(zero_x_inds)-2, 2):
                    idx0 = zero_x_inds[zx_ind]
                    idx1 = zero_x_inds[zx_ind+1]
                    idx2 = zero_x_inds[zx_ind+2]
                    if (idx1 - idx0) < min_samples or (idx2 - idx1) < min_samples:
                        continue
                    time0 = raw_work.first_time + raw_work.times[idx0]
                    time1 = raw_work.first_time + raw_work.times[idx2]
                    peak_time_idx = np.min(find_peaks(signal[idx1:idx2])[0]) + idx1
                    trough_time_idx = np.argmin(signal[idx0:idx1]) + idx0
                    peak_amp, trough_amp = signal[peak_time_idx], signal[trough_time_idx]
                    peak_time = raw_work.first_time + raw_work.times[peak_time_idx]
                    trough_time = raw_work.first_time + raw_work.times[trough_time_idx]
                    osc_events.append(OscEvent(time0, time1, peak_time,
                                               peak_amp, trough_time, trough_amp))
                # get percentiles of peaks and troughs
                peaks, troughs = osc_peaktroughs(osc_events)
                amps = peaks - troughs
                this_df = df.query("Cond!='sham' and Subj=='{}' and OscType=='{}' and Chan=='{}'".format(subj, osc_type, k))
                amp_thresh = np.mean(this_df["Thresh"].values)
                mark_osc_amp(osc_events, amp_thresh, k, minmax_time, osc_type)
                marked_oe = [oe for oe in osc_events if oe.event_id is not None]
                for moe_idx, moe in enumerate(marked_oe):
                    if moe_idx == 0:
                        new_annots = mne.Annotations(moe.start_time,
                                                     moe.end_time-moe.start_time,
                                                     "{} {}".format(moe.event_id, moe.event_annot),
                                                     orig_time=raw_work.annotations.orig_time)
                    else:
                        new_annots.append(moe.start_time, moe.end_time-moe.start_time,
                                          "{} {}".format(moe.event_id, moe.event_annot))
                    new_annots.append(moe.trough_time, 0,
                                      "Trough {} {}".format(moe.event_id, moe.event_annot))
                    new_annots.append(moe.peak_time, 0,
                                      "Peak {} {}".format(moe.event_id, moe.event_annot))


                raw_work.set_annotations(new_annots)
                raw.set_annotations(new_annots)
                start_stim = find_densest(new_annots, 1800)
                start_dict["Subj"].append(subj)
                start_dict["OscType"].append(osc_type)
                start_dict["Chan"].append(k)
                start_dict["Start"].append(start_stim)

df = pd.DataFrame.from_dict(start_dict)
df.to_pickle("{}start_times.pickle".format(proc_dir))
