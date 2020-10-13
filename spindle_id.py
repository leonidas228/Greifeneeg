import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
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
n_jobs = 8
spindle_freq = np.arange(10,15)
spindle_time = 0.35
gw_time = spindle_time/2
#conds = ["sham"]
spindle_thresh = (1.5, 2.5)

for filename in filelist:
    this_match = re.match("aibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
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

        upper_thresh = tfr_con.mean() + spindle_thresh[1]*tfr_con.std()
        lower_thresh = tfr_con.mean() + spindle_thresh[0]*tfr_con.std()
        tfr_thresh = tfr_con - lower_thresh
        # need to add infinitesimals to zeros to prevent weird x-crossing bugs
        for null_idx in list(np.where(tfr_thresh==0)[0]):
            if null_idx:
                tfr_thresh[null_idx] = 1e-16*np.sign(tfr_thresh[null_idx-1])
            else:
                tfr_thresh[null_idx] = 1e-16*np.sign(tfr_thresh[null_idx+1])
        tfr_xcross = np.where((tfr_thresh[1:] * tfr_thresh[:-1]<0))[0]
        x_start = 0 if tfr_thresh[0] < 0 else 1
        spindles = []
        spindle_time_ind = int(raw_work.info["sfreq"] * spindle_time)
        spindle_idx = 0
        for xc in range(x_start, len(tfr_xcross), 2):
            if xc+1 >= len(tfr_xcross):
                continue # recording ended on a possible spindle; ignore
            if (tfr_xcross[xc+1] - tfr_xcross[xc]) < spindle_time_ind:
                continue
            if tfr_con[tfr_xcross[xc]:tfr_xcross[xc+1]].max() < upper_thresh:
                continue
            raw_work.annotations.append(ft+raw_work.times[tfr_xcross[xc]],
                                        raw_work.times[tfr_xcross[xc+1]] - raw_work.times[tfr_xcross[xc]],
                                        "Spindle {}".format(spindle_idx))
            spindle_peak = raw_work.times[np.argmax(tfr_con[tfr_xcross[xc]:tfr_xcross[xc+1]]) + tfr_xcross[xc]]
            raw_work.annotations.append(ft+spindle_peak, 0, "Spindle peak {}".format(spindle_idx))
            spindle_idx += 1
            print("Spindle at {}".format(raw_work.times[tfr_xcross[xc]]))
        raw.set_annotations(raw_work.annotations)
        raw.save("{}aaibscaf_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond),
                 overwrite=True)
        # raw_work.set_annotations(annot_subset(raw_work.annotations, "Spindle"))
        # raw_work.plot(scalings="auto")
