import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
plt.ion()

def annot_subset(annotations, match_str):
    delete_inds = []
    for ann_idx, ann in enumerate(annotations):
        match = re.search(match_str, ann["description"])
        if not match:
            delete_inds.append(ann_idx)
    annotations.delete(delete_inds)
    return annotations

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)

channel = "central"
osc = "SO"
n_jobs = 8
spindle_freq = np.arange(10,20)
spindle_time = 0.35
gw_time = spindle_time/2
#conds = ["sham"]
spindle_thresh = (1.5, 2.5)
spindle_thresh = (1.3, 2)

for filename in filelist:
    this_match = re.match("NAP_(\d{3})_(.*)_(.*)_(.*)-raw.fif",filename)
    if this_match:
        subj, cond, ort, this_osc = this_match.group(1), this_match.group(2), this_match.group(3), this_match.group(4)
        if ort != channel or this_osc != osc:
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        raw_work = raw.copy()
        ft = raw_work.first_time
        raw_work.pick_channels([channel])
        raw_work.filter(l_freq=spindle_freq[0], h_freq=spindle_freq[1], n_jobs=n_jobs)
        epo = mne.make_fixed_length_epochs(raw_work, duration=raw_work.times[-1])
        power = tfr_morlet(epo, spindle_freq, n_cycles=5, average=False,
                           return_itc=False, n_jobs=n_jobs)

        tfr = np.zeros(0)
        for epo_tfr in power.__iter__():
            tfr = np.concatenate((tfr,np.mean(epo_tfr[:,0,],axis=0)))
        gw_len = np.round(gw_time * raw.info["sfreq"]).astype(int)
        gauss_win = np.exp(-0.5*((np.arange(gw_len)-gw_len/2)/(0.5*gw_len/2))**2)
        #tfr_con = np.convolve(tfr, gauss_win, mode="same")
        tfr_con = tfr.copy()

        tfr_aschan = np.zeros(len(raw_work))
        tfr_aschan[:len(tfr)] = tfr_con * 10
        tfr_raw = mne.io.RawArray(np.expand_dims(tfr_aschan,0), mne.create_info(["TFR"],raw_work.info["sfreq"],ch_types="misc"))
        raw_work.add_channels([tfr_raw], force_update_info=True)
        raw.add_channels([tfr_raw], force_update_info=True)

        widths = [int(raw_work.info["sfreq"]*spindle_time)]
        peaks = find_peaks_cwt(tfr_con, widths)
        peak_times = list(raw_work.times[peaks])
        for peak_idx, peak in enumerate(peak_times):
            raw_work.annotations.append(ft+peak, 0, "Spindle peak {}".format(peak_idx))

        raw.set_annotations(raw_work.annotations)
        raw.save("{}spind_{}_NAP_{}_{}_{}-raw.fif".format(proc_dir,subj,cond,ort,osc),
                 overwrite=True)

        # raw_work.set_annotations(annot_subset(raw_work.annotations, "Spindle"))
        # raw_work.plot(scalings="auto")
        # breakpoint()
