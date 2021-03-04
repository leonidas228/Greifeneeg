import mne
from os import listdir
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from os.path import isdir
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

def mark_osc_amp(osc_events, amp_thresh, chan_name, mm_times, osc_type,
                 raw_inst=None):
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

eig_freqs = {'002': 0.8135739656, '003': 0.8388119505, '005': 0.6697439105,
             '006': 0.7832908796, '007': 0.7456094526, '009': 0.7788409575,
             '013': 0.6714744633, '015': 0.8900342154, '016': 0.8195879177,
             '017': 0.807011053, '018': 0.8468273959, '021': 0.5211960758,
             '022': 0.5030628337, '024': 0.5370360281, '025': 0.5080682169,
             '026': 0.511158806, '027': 0.5085371744, '028': 0.6724768668,
             '031': 0.8641548566, '033': 0.9979009669, '035': 0.6158327557,
             '037': 0.6919189534, '038': 0.6330339807, '043': 0.6852392211,
             '044': 0.6559392045, '045': 0.6264749245, '046': 0.6615336116,
             '047': 0.8233948546, '048': 0.7762510701, '050': 0.6685955766,
             '051': 0.7523264618, '053': 0.8741048686, '054': 0.6611531633}

ages = {"002":55, "003":66, "005":77, "006":52, "007":65, "009":71, "013":76,
        "015":56, "016":51, "017":66, "018":50, "021":77, "022":55, "024":66,
        "025":57, "026":63, "027":71, "028":69, "031":57, "033":56, "035":69,
        "037":55, "038":74, "043":77, "044":68, "045":73, "046":67, "047":62,
        "048":70, "050":70, "051":76, "053":63, "054":71}

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham30s", "sham2m", "sham5m"]
filelist = listdir(proc_dir)
# chan_groups = {"frontal":["Fz","FC1","FC2"],
#                "central":["Cz","CP1","CP2"]}
chan_groups = {"central":["Fz","FC1","FC2", "Cz","CP1","CP2"]}
amp_percentile = 65
min_samples = 10
minmax_freqs = [(0.16, 1.25), (0.75, 4.25)]
minmax_times = [(0.8, 2), (0.25, 1)]
osc_types = ["SO", "deltO"]
includes = []
amp_thresh_dict = {"Subj":[], "Cond":[], "OscType":[], "Chan":[], "Thresh":[]}
skipped = []

for filename in filelist:
    this_match = re.match("ibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        if len(includes) and "{}_{}".format(subj,cond) not in includes:
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)

        # crop at certain time
        for annot in raw.annotations:
            if annot["description"] == "Post_Stimulation 0":
                break
        raw.crop(tmin=(annot["onset"] - raw.first_samp / raw.info["sfreq"]))

        # produce channel-ROI averages
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
            raw_work.filter(l_freq=minmax_freq[0], h_freq=minmax_freq[1])
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
                osc_events = [oe for oe in osc_events if (oe.end_time-oe.start_time)>minmax_time[0] and (oe.end_time-oe.start_time)<minmax_time[1]]
                peaks, troughs = osc_peaktroughs(osc_events)
                amps = peaks - troughs
                amp_thresh = np.percentile(amps, amp_percentile)
                amp_thresh_dict["Subj"].append(subj)
                amp_thresh_dict["Cond"].append(cond)
                amp_thresh_dict["OscType"].append(osc_type)
                amp_thresh_dict["Chan"].append(k)
                amp_thresh_dict["Thresh"].append(amp_thresh)

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
                    new_annots.save("{}NAP_{}_{}_{}_{}-annot.fif".format(proc_dir,subj,cond,k,osc_type))
                    raw.set_annotations(new_annots)
                else:
                    skipped.append("{} {} {} {}".format(subj, cond, k, osc_type))
                    print("\nNo oscillations found. Skipping.\n")
                    continue

                events = mne.events_from_annotations(raw, check_trough_annot)
                df_dict = {"Subj":[],"Cond":[],"StimType":[],"Dur":[],
                           "PrePost":[],"Index":[],"Stim":[],"PureIndex":[],
                           "OscType":[], "Sync":[], "OscLen":[],"OscFreq":[],
                           "EigFreq":[], "Age":[]}
                for event_idx, event in enumerate(np.nditer(events[0][:,-1])):
                    eve = event.copy()
                    if eve >= 100:
                        df_dict["PureIndex"].append(str(eve//100))
                        eve = eve%100
                    else:
                        df_dict["PureIndex"].append(str(0))
                    if eve >= 50:
                        df_dict["PrePost"].append("Post")
                        eve -= 50
                    else:
                        df_dict["PrePost"].append("Pre")
                    df_dict["Index"].append(int(eve))
                    df_dict["Subj"].append(subj)
                    if int(subj) < 31 and subj != "021" and subj!='017':
                        df_dict["Sync"].append("async")
                    else:
                        df_dict["Sync"].append("sync")

                    df_dict["Cond"].append(cond)
                    if "eig" in cond:
                        df_dict["StimType"].append("eig")
                    elif "fix" in cond:
                        df_dict["StimType"].append("fix")
                    elif "sham" in cond:
                        df_dict["StimType"].append("sham")
                    if "30s" in cond:
                        df_dict["Dur"].append("30s")
                    elif "2m" in cond:
                        df_dict["Dur"].append("2m")
                    elif "5m" in cond:
                        df_dict["Dur"].append("5m")

                    if "sham" not in cond:
                        df_dict["Stim"].append("stim")
                    else:
                        df_dict["Stim"].append("sham")
                    df_dict["OscType"].append(osc_type)

                    df_dict["OscLen"].append(marked_oe[event_idx].end_time - marked_oe[event_idx].start_time)
                    df_dict["OscFreq"].append(1/df_dict["OscLen"][-1])
                    df_dict["EigFreq"].append(eig_freqs[subj])
                    df_dict["Age"].append(ages[subj])

                df = pd.DataFrame.from_dict(df_dict)
                epo = mne.Epochs(raw, events[0], tmin=-2.5, tmax=2.5, detrend=1,
                                 baseline=None, metadata=df, event_repeated="drop").load_data()
                raw.save("{}NAP_{}_{}_{}_{}-raw.fif".format(proc_dir,subj,cond,k,osc_type),
                         overwrite=True)
                epo.save("{}NAP_{}_{}_{}_{}-epo.fif".format(proc_dir,subj,cond,k,osc_type),
                         overwrite=True)

df = pd.DataFrame.from_dict(amp_thresh_dict)
df.to_pickle("{}amp_thresh.pickle".format(proc_dir))
