import mne
from os import listdir
import re
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

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
        event = 0
        if "Central" in desc:
            event += 100
        if "Post" in desc:
            event += 50
        event += int(desc[-1])
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

def mark_osc(osc_events, amp_thresh, trough_thresh, sig_name, raw_inst=None):
    oe_idx = 0
    for oe in osc_events:
        pt_time_diff = oe.trough_time - oe.peak_time
        if (oe.trough_amp < trough_thresh and
            (oe.peak_amp - oe.trough_amp) > amp_thresh):
            if raw_inst is not None:
                event_annot = get_annotation(raw_inst.annotations,
                                                oe.start_time)
                if event_annot is None:
                    continue
            else:
                event_annot = None
            # is a slow oscillation
            oe.event_id = "{} SO {}".format(sig_name, oe_idx)
            oe.event_annot = event_annot
            oe_idx += 1


root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
filelist = listdir(proc_dir)
chan_groups = {"central":["Fz","FC1","FC2","Cz","CP1","CP2","Pz"]}
amp_percentile = 0.75
trough_thresh = -2e-5
trough_thresh = np.inf
so_time_range = (.15,.5)
delta_time_range = 500

for filename in filelist:
    this_match = re.match("ibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        # produce channel-ROI averages
        for k,v in chan_groups.items():
            raw_pick = raw.copy().pick_channels(v)
            avg_signal = raw_pick.get_data().mean(axis=0, keepdims=True)
            avg_info = mne.create_info([k], raw.info["sfreq"], ch_types="eeg")
            avg_raw = mne.io.RawArray(avg_signal, avg_info)
            raw.add_channels([avg_raw], force_update_info=True)
        raw_work = raw.copy()
        raw_work.filter(l_freq=0.16, h_freq=1.25)
        first_time = raw_work.first_samp / raw_work.info["sfreq"]

        # zero crossings
        zero_xs = {}
        for k in chan_groups.keys():
            pick_ind = mne.pick_channels(raw_work.ch_names, include=[k])
            signal = raw_work.get_data()[pick_ind,].squeeze()
            zero_x_inds = (np.where((signal[:-1] * signal[1:]) < 0)[0]) + 1

            # cycle through negative crossings
            neg_x0_ind = 1 if signal[0] < 0 else 0
            osc_events = []
            for zx_ind in range(neg_x0_ind, len(zero_x_inds)-2, 2):
                idx0 = zero_x_inds[zx_ind]
                idx1 = zero_x_inds[zx_ind+2]
                time0 = raw_work.first_time + raw_work.times[idx0]
                time1 = raw_work.first_time + raw_work.times[idx1]
                duration = time1 - time0
                if duration > 0.8 and duration < 2:
                    peak_time_idx = np.argmax(signal[idx0:idx1]) + idx0
                    trough_time_idx = np.argmin(signal[idx0:idx1]) + idx0
                    peak_amp, trough_amp = signal[peak_time_idx], signal[trough_time_idx]
                    peak_time = raw_work.first_time + raw_work.times[peak_time_idx]
                    trough_time = raw_work.first_time + raw_work.times[trough_time_idx]
                    osc_events.append(OscEvent(time0, time1, peak_time,
                                               peak_amp, trough_time, trough_amp))
            # get percentiles of peaks and troughs
            peaks, troughs = osc_peaktroughs(osc_events)
            amps = peaks - troughs
            amp_thresh = np.percentile(amps, amp_percentile)

            mark_osc(osc_events, amp_thresh, trough_thresh, k,
                     raw_inst=raw_work)
            marked_oe = [oe for oe in osc_events if oe.event_id is not None]
            for moe_idx, moe in enumerate(marked_oe):
                if moe_idx == 0:
                    new_annots = mne.Annotations(moe.start_time,
                                                 moe.end_time-moe.start_time,
                                                 "{} {}".format(moe.event_id, moe.event_annot),
                                                 orig_time=raw_work.annotations.orig_time)
                    raw_work.set_annotations(new_annots)
                else:
                    raw_work.annotations.append(moe.start_time, moe.end_time-moe.start_time,
                                                "{} {}".format(moe.event_id, moe.event_annot))
                raw_work.annotations.append(moe.trough_time, 0,
                                            "Trough {} {}".format(moe.event_id, moe.event_annot))

        raw.set_annotations(raw_work.annotations)
        raw.save("{}aibscaf_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond),
                 overwrite=True)
        events = mne.events_from_annotations(raw,check_trough_annot)
        epo = mne.Epochs(raw,events[0], events[1], tmin=-1.25, tmax=1.25,
                         baseline=(None,None))
        epo.save("{}NAP_{}_{}-epo.fif".format(proc_dir,subj,cond), overwrite=True)
