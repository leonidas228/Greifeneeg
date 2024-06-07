import mne
from os import listdir,getcwd
from os.path import isdir, join
import re
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()

def annot_stim(ur_raw, tfr_thresh_range = list(np.linspace(0.001,0.01,100)),
    tfr_lower_thresh = 1e-6,
    pre_stim_buffer = 20,
    post_stim_buffer = 30,
    analy_duration = 60,
    between_duration = 60,
    epolen = 10,
    min_bad = 25,
    picks = None, #["Fz","AFz","Fp1","Fp2","FC1","FC2","Cz"]
    n_jobs = 24,
    post_only = True,
    exclude = ["1038", "1026", "1036"],
    overwrite = True
):
    raw = ur_raw.copy()
    spectrum = raw.compute_psd(method="multitaper", fmax=2, picks=picks, n_jobs=n_jobs)
    #raw.plot_psd(method='multitaper', fmax=2, picks = 'Fz')

    psd = spectrum.get_data().mean(axis=0)
    fmax = spectrum.freqs[np.argmax(psd)]
    fmax = 0.8
    #plt.plot(spectrum.freqs, psd)
    #plt.show(block=True)
    epo = mne.make_fixed_length_epochs(raw, duration=epolen)
    power = epo.compute_tfr(method='morlet', freqs=[fmax], n_cycles=5, picks=picks,
                        average=False, return_itc=False, n_jobs=n_jobs)
    #power.plot(0)
    tfr = np.zeros(0)
    for epo_tfr in power.__iter__():
        tfr = np.concatenate((tfr,np.mean(epo_tfr[:,0,],axis=0)))
    tfr_aschan = np.zeros(len(raw))
    tfr_aschan[:len(tfr)] = tfr
    
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
                                            "BAD_Stimulation_{}".format(stim_idx))
                if not stim_idx:
                    these_annotations.append(begin - pre_dur, pre_dur,
                                                "Pre_Stimulation_{}".format(stim_idx))
                these_annotations.append(begin + duration, post_dur,
                                            "Post_Stimulation_{}".format(stim_idx))
                earliest_idx = idx
                stim_idx += 1
            else:
                these_annotations.append(begin, duration,
                                            "BAD_Stimulation_{}".format(stim_idx))
                these_annotations.append(begin - pre_dur, pre_dur,
                                            "Pre_Stimulation_{}".format(stim_idx))
                these_annotations.append(begin + duration, post_dur,
                                            "Post_Stimulation_{}".format(stim_idx))
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
    if winner_std > 2:
        breakpoint()

    return (raw,winner_annot)


if __name__ == '__main__':
    root_dir = getcwd()
    proc_dir = join(root_dir, "data/proc")
    tfr_thresh_range = list(np.linspace(0.001,0.01,100))
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
    n_jobs = 24
    post_only = True
    exclude = ["1038", "1026", "1036"]
    overwrite = True

    filenames = listdir(proc_dir)
    for filename in filenames:
        match = re.match("p_NAP_(\d{4})_(.*)-raw.fif", filename)
        if not match:
            continue
        (subj, cond) = match.groups()
        if cond == "sham" or subj in exclude:
            continue

        outname = f"stim_NAP_{subj}_{cond}-annot.fif"
        if outname in filenames and not overwrite:
            print(f"{outname} already exists. Skipping...")
            continue
        
        ur_raw = mne.io.Raw(join(proc_dir, filename), preload=True)
        res = annot_stim(ur_raw)
        print(res)
        raw = res[0]
        annot = res[1]
        annot.save(join(proc_dir, outname), overwrite=overwrite)
        raw.plot(block=True)
        break 
