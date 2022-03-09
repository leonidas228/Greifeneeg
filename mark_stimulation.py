import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis
from mne.time_frequency import psd_multitaper, tfr_morlet
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"

proc_dir = root_dir+"proc/"
tfr_thresh_range = list(np.linspace(0.001,0.008,50))
tfr_lower_thresh = 1e-6
pre_stim_buffer = 20
post_stim_buffer = 30
analy_duration = 60
between_duration = 60
filelist = listdir(proc_dir)
epolen = 10
min_bad = 25
picks = ["Fz","AFz","Fp1","Fp2","FC1","FC2","Cz"]
dur_dict = {344:"5m", 165:"2m", 73:"30s"}
n_jobs = 8
post_only = True
do_subj = "043"

with open("stim_info.csv", "wt") as f:
    f.write("Subject\tBlock\tBegin\tEnd\tLength\tTotal\n")
with open("randomisierung.csv", "wt") as f:
    f.write("Subject\tDay\tBlock\tFrequency\tTotal\n")

for filename in filelist:
    this_match = re.match("f_NAP_(\d{3})_T(\d)-raw.fif",filename)
    if this_match:
        subj, tag = this_match.group(1), int(this_match.group(2))
        # if subj != do_subj:
        #     continue
        if tag < 2:
            raw = mne.io.Raw(proc_dir+filename,preload=True)
            raw.save("{}af_NAP_{}_{}-raw.fif".format(proc_dir, subj, tag),
                     overwrite=True)
            continue

        ur_raw = mne.io.Raw(proc_dir+filename,preload=True)
        raw = ur_raw.copy()
        psds, freqs = psd_multitaper(raw, fmax=2, picks=picks, n_jobs=n_jobs)
        psds = psds.mean(axis=0)
        fmax = freqs[np.argmax(psds)]
        print("Max freqeuncy: {}".format(fmax))
        if fmax < 0.76 and fmax > 0.74:
            stim_type = "fix"
        elif fmax < 0.3:
            stim_type = "sham"
            stim_len = ""
        else:
            stim_type = "eig"

        # special cases
        if subj == "007" or subj == "051":
            if stim_type == "eig":
                stim_type = "fix"
            elif stim_type == "fix":
                stim_type = "eig"

        if stim_type != "sham":
            epo = mne.make_fixed_length_epochs(raw, duration=epolen)
            #epo = mne.Epochs(raw, events, tmin=0, tmax=epolen, baseline=None)

            power = tfr_morlet(epo, [fmax], n_cycles=3, picks=picks,
                               average=False, return_itc=False, n_jobs=n_jobs)

            tfr = np.zeros(0)
            for epo_tfr in power.__iter__():
                tfr = np.concatenate((tfr,np.mean(epo_tfr[:,0,],axis=0)))
            tfr_aschan = np.zeros(len(raw))
            tfr_aschan[:len(tfr)] = tfr

            print("\n{} - {}\n".format(subj, tag))
            winner_std = np.inf
            for tfr_upper_thresh in tfr_thresh_range:
                these_annotations = raw.annotations.copy()
                tfr_over_thresh = (tfr_aschan > tfr_upper_thresh).astype(float) - 0.5
                tfr_over_cross = tfr_over_thresh[:-1] * tfr_over_thresh[1:]
                tfr_over_cross = np.concatenate((np.zeros(1),tfr_over_cross))
                tfr_under_thresh = (tfr_aschan < tfr_lower_thresh).astype(float) - 0.5
                tfr_under_cross = tfr_under_thresh[:-1] * tfr_under_thresh[1:]
                tfr_under_cross = np.concatenate((np.zeros(1),tfr_under_cross))
                tfr_under_cross_inds = np.where(tfr_under_cross < 0)[0]

                if (len(np.where(tfr_over_cross < 0)[0]) == 0 or
                    len(np.where(tfr_over_cross < 0)[0]) == 0):
                    continue

                earliest_idx = 0
                stim_idx = 0
                for cross in np.nditer(np.where(tfr_over_cross < 0)[0]):
                    if cross < earliest_idx:
                        continue
                    min_bad_idx = cross + int(np.round(min_bad * raw.info["sfreq"]))
                    if min_bad_idx > len(tfr_under_thresh):
                        min_bad_idx = len(tfr_under_thresh) - 1
                    if tfr_under_thresh[min_bad_idx] > 0: # false alarm; do not mark
                        earliest_idx = min_bad_idx
                        continue

                    begin = raw.times[cross] - pre_stim_buffer
                    idx = tfr_under_cross_inds[tfr_under_cross_inds > min_bad_idx][0]
                    end = raw.times[idx] + post_stim_buffer
                    duration = end - begin
                    if stim_idx == 0:
                        pre_dur = analy_duration
                        post_dur = between_duration
                    else:
                        pre_dur = between_duration
                        post_dur = between_duration
                    if post_only:
                        these_annotations.append(begin, duration,
                                                 "BAD_Stimulation {}".format(stim_idx))
                        if not stim_idx:
                            these_annotations.append(begin - pre_dur, pre_dur,
                                                     "Pre_Stimulation {}".format(stim_idx))
                        these_annotations.append(begin + duration, post_dur,
                                                 "Post_Stimulation {}".format(stim_idx))
                        earliest_idx = idx
                        stim_idx += 1
                    else:
                        these_annotations.append(begin, duration,
                                                 "BAD_Stimulation {}".format(stim_idx))
                        these_annotations.append(begin - pre_dur, pre_dur,
                                                 "Pre_Stimulation {}".format(stim_idx))
                        these_annotations.append(begin + duration, post_dur,
                                                 "Post_Stimulation {}".format(stim_idx))
                        earliest_idx = idx
                        stim_idx += 1

                # assess uniformity
                durations = []
                for annot in these_annotations:
                    if "BAD" in annot["description"]:
                        durations.append(annot["duration"])
                dur_std = np.array(durations).std()
                if dur_std < winner_std and dur_std != 0.:
                    winner_annot = these_annotations.copy()
                    winner_std =  dur_std
                    winner_id = tfr_upper_thresh
                    winner_stim_idx = stim_idx
                    winner_durations = durations.copy()

            # last post-stimulation period should be longer
            last_annot = winner_annot[-1].copy()
            winner_annot.delete(-1)
            winner_annot.append(last_annot["onset"], analy_duration, last_annot["description"])

            raw.set_annotations(winner_annot)
            print("\nThreshold of {} was optimal.\nDurations:".format(winner_id))
            print(winner_durations)
            print("\nStd:{}\n".format(winner_std))

            avg_dur = np.array(winner_durations).mean()
            dur_refs = np.array(list(dur_dict.keys()))
            this_dur = dur_refs[np.argmin(abs(dur_refs-avg_dur))]
            stim_len = dur_dict[this_dur]

            with open("stim_info.csv", "at") as f:
                for si in range(winner_stim_idx):
                    for annot in winner_annot:
                        if annot["description"] == "BAD_Stimulation {}".format(si):
                            break
                    begin = annot["onset"]
                    duration = annot["duration"]
                    end = begin + duration
                    f.write("{}\t{}\t{}\t{}\t{}\t".format(subj, stim_type+stim_len, int(begin),
                                                          int(end), int(duration)))
                    if si == winner_stim_idx - 1:
                        f.write("{}\n".format(winner_stim_idx))
                    else:
                        f.write("\n")

        with open("randomisierung.csv", "at") as f:
            if stim_type == "sham":
                f.write("{}\t{}\t{}\t\t{}\n".format(subj, tag, stim_type, ""))
            else:
                f.write("{}\t{}\t{}\t{}\t{}\n".format(subj, tag, stim_type+stim_len, fmax, winner_stim_idx))

        if stim_type == "sham":
            raw.save("{}f_NAP_{}_{}{}-raw.fif".format(proc_dir,subj,stim_type,stim_len),
                     overwrite=True)
        else:
            raw.save("{}af_NAP_{}_{}{}-raw.fif".format(proc_dir,subj,stim_type,stim_len),
                     overwrite=True)
