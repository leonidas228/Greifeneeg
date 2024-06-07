import mne
from os import listdir, getcwd
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from anoar import BadChannelFind
from scipy.signal import find_peaks
from os.path import isdir, join
plt.ion()

"""
Mark slow and delta oscillations in data and save as raw and epoched formats.
"""


class OscEvent():
    # organising class for oscillatory events
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
    # helper function for marking troughs of oscillations
    event = None
    if "Trough" in desc:
        event = int(desc[-1])
    return event

def get_annotation(annotations, time):
    # does a time period reside in a Post stim annotation?
    period = None
    for annot in annotations:
        if "Post" not in annot["description"]:
            continue
        begin = annot["onset"]
        end = begin + annot["duration"]
        if time > begin and time < end:
            period = annot["description"]
    return period

def osc_peaktroughs(osc_events):
    # get peaks and troughs of an OscEvent instance
    peaks = []
    troughs = []
    for oe in osc_events:
        peaks.append(oe.peak_amp)
        troughs.append(oe.trough_amp)
    peaks, troughs = np.array(peaks), np.array(troughs)
    return peaks, troughs

def mark_osc_amp(osc_events, amp_thresh, chan_name, mm_times, osc_type,
                 raw_inst=None):
    # 
    osc_idx = 0
    for oe in osc_events:
        if raw_inst is not None:
            event_annot = get_annotation(raw_inst.annotations,
                                         oe.start_time)
            if event_annot is None:
                continue
        else:
            event_annot = None
        pt_time_diff = oe.trough_time - oe.peak_time
        time_diff = oe.end_time - oe.start_time
        pt_amp_diff = oe.peak_amp - oe.trough_amp
        if pt_amp_diff > amp_thresh and mm_times[0] < time_diff < mm_times[1]:
            oe.event_id = "{} {} {}".format(chan_name, osc_type, osc_idx)
            oe.event_annot = event_annot
            osc_idx += 1



chan_groups = {"frontal":["Fz", "FC1", "FC2"],
               "parietal":["Cz", "CP1", "CP2"]}
amp_percentile = 65
min_samples = 10
minmax_freqs = [(0.16, 1.25), (0.75, 4.25)]
minmax_times = [(0.8, 2), (0.25, 1)]
osc_types = ["SO"]
includes = []
skipped = {"no_osc":[], "few_osc":[], "chan":[], "ROI":[]}

root_dir = getcwd()
proc_dir = join(root_dir, "data/proc")

gap_dict = {
            "1001":{"anodal":16, "cathodal":1060},
            "1002":{"anodal":14, "cathodal":535},
            "1003":{"anodal":14, "cathodal":994},
            "1004":{"anodal":7, "cathodal":745},
            "1005":{"anodal":7, "cathodal":1141},
            "1006":{"anodal":41, "cathodal":859},
            "1008":{"anodal":7, "cathodal":906},
            "1011":{"anodal":7, "cathodal":995},
            "1012":{"anodal":7, "cathodal":695},
            "1013":{"anodal":14, "cathodal":798},
            "1020":{"anodal":7, "cathodal":42},
            "1021":{"anodal":8, "cathodal":28},
            "1023":{"anodal":11, "cathodal":1147},
            "1026":{"anodal":21, "cathodal":52},
            "1036":{"anodal":23, "cathodal":76},
            "1038":{"anodal":22, "cathodal":42},
            "1042":{"anodal":18, "cathodal":25}
}

filelist = listdir(proc_dir)
for filename in filelist:
    this_match = re.match("cp_NAP_(\d{4})_(.*)-raw.fif", filename)
    if not this_match:
        continue
    (subj, cond) = this_match.groups()

    c_groups = chan_groups.copy()
    # special case
    if subj == "1021":
        if cond == "sham" or cond == "anodal":
            c_groups["parietal"] = ["P1", "P2", "CPz"]
            c_groups["frontal"] = ["F1", "Iz", "F2"]

    raw = mne.io.Raw(join(proc_dir, filename), preload=True)
    picks = mne.pick_types(raw.info, eeg=True)

    passed = np.zeros(len(c_groups), dtype=bool)
    for idx, (k,v) in enumerate(c_groups.items()):
        pick_list = [vv for vv in v if vv not in raw.info["bads"]]
        if not len(pick_list):
            print("No valid channels")
            skipped["chan"].append("{} {}".format(subj, cond))
            continue
        avg_signal = raw.get_data(pick_list).mean(axis=0, keepdims=True)
        avg_info = mne.create_info([k], raw.info["sfreq"], ch_types="eeg")
        avg_raw = mne.io.RawArray(avg_signal, avg_info)
        raw.add_channels([avg_raw], force_update_info=True)
        passed[idx] = 1
    if not all(passed):
        print("Could not produce valid ROIs")
        skipped["ROI"].append("{} {}".format(subj, cond))
        continue
    # ROIs only, drop everything els
    raw.pick_channels(list(chan_groups.keys()))

    # sort out conditiona and polarity
    if cond == "sfb1":
        cond = "sham"
        polarity = "cathodal"
    elif cond == "sham":
        polarity = "anodal"
    elif cond == "anodal":
        cond = "stim"
        polarity = "anodal"
    elif cond == "cathodal":
        cond = "stim"
        polarity = "cathodal"
    else:
        raise ValueError("Could not organise condition/polarity")
    # = gap_dict[subj][polarity]
    

    for minmax_freq, minmax_time, osc_type in zip(minmax_freqs, minmax_times, osc_types):
        raw_work = raw.copy()
        raw_work.filter(l_freq=minmax_freq[0], h_freq=minmax_freq[1])
        first_time = raw_work.first_samp / raw_work.info["sfreq"]

        # zero crossings
        for k in raw.ch_names:
            df_dict = {"Subj":[],"Cond":[],"Index":[], "ROI":[], "Polarity":[],
                       "OscType":[], "OscLen":[], "OscFreq":[]}
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
            osc_events = [oe for oe in osc_events if (oe.end_time-oe.start_time)>minmax_time[0] and 
                          (oe.end_time-oe.start_time)<minmax_time[1]]
            peaks, troughs = osc_peaktroughs(osc_events)
            amps = peaks - troughs
            amp_thresh = np.percentile(amps, amp_percentile)

            mark_osc_amp(osc_events, amp_thresh, k, minmax_time, osc_type,
                         raw_inst=raw_work)
            marked_oe = [oe for oe in osc_events if oe.event_id is not None]
            if len(marked_oe):
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
                new_annots.save(join(proc_dir, 
                                     f"osc_NAP_{subj}_{cond}_{k}_{osc_type}_{polarity}-annot.fif"),
                                overwrite=True)
                raw.set_annotations(new_annots)
            else:
                skipped["no_osc"].append("{} {} {} {} {}".format(subj, cond, k, osc_type, polarity))
                print("\nNo oscillations found. Skipping.\n")
                continue

            events = mne.events_from_annotations(raw, check_trough_annot)

            for event_idx, event in enumerate(events[0][:,-1]):
                eve = event.copy()
                df_dict["Index"].append(int(eve))
                df_dict["Subj"].append(subj)
                df_dict["Cond"].append(cond)
                df_dict["Polarity"].append(polarity)
                #df_dict["Gap"].append(gap)
                df_dict["ROI"].append(k)
                df_dict["OscType"].append(osc_type)
                df_dict["OscLen"].append(marked_oe[event_idx].end_time - marked_oe[event_idx].start_time)
                df_dict["OscFreq"].append(1/df_dict["OscLen"][-1])

            df = pd.DataFrame.from_dict(df_dict)
            epo = mne.Epochs(raw, events[0], tmin=-2.5, tmax=2.5, detrend=1,
                             baseline=None, metadata=df, event_repeated="drop",
                             reject={"eeg":5e-4}).load_data()
                
            if len(epo) < 25:
                skipped["few_osc"].append("{} {} {} {} {}".format(subj, cond, k, osc_type, polarity))
                continue
            raw.save(join(proc_dir, f"osc_NAP_{subj}_{cond}_{k}_{osc_type}_{polarity}-raw.fif"),
                     overwrite=True)
            epo.save(join(proc_dir, f"osc_NAP_{subj}_{cond}_{k}_{osc_type}_{polarity}-epo.fif"),
                     overwrite=True)

with open(join(proc_dir, "skipped_record.pickle"), "wb") as f:
    pickle.dump(skipped, f)
