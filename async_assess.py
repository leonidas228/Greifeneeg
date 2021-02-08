import mne
from anoar import BadChannelFind
from os import listdir
import re
from os.path import isdir
import numpy as np
from mne.time_frequency import psd_multitaper, tfr_morlet
import matplotlib.pyplot as plt
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s"]
filelist = listdir(proc_dir)
msqrt_thresh = {"5m_async":1.5, "5m_sync":0.2, "2m_async":0.5, "2m_sync":0.5,
                "30s_sync":0.5, "30s_async":0.5}
n_jobs = 8
durs = ["30s","2m","5m"]
syncs = ["async", "sync"]
left_chans = ["FC5", "FC1", "C3", "CP1", "T7", "P3"]
right_chans = ["FC6", "FC2", "C6", "CP2", "T8", "P4"]

excludes = ['015_eig30s', '025_fix30s', '015_fix30s', '027_eig30s',
            '016_fix30s', '027_fix30s', '033_eig30s', '053_fix30s',
            '045_fix30s', '025_eig2m', '017_fix2m', '044_fix2m', '022_eig5m',
            '031_eig5m', '048_eig5m', '046_eig5m', '035_eig5m', '044_eig5m',
            '053_eig5m']
excludes = []

bad_stims = []
for dur in durs:
    for sync in syncs:
        power_list = []
        phase_list = []
        fmax_list = []
        subid_list = []
        for filename in filelist:
            this_match = re.match("bad_caf_NAP_(\d{3})_(.*)-raw.fif",filename)
            if this_match:
                this_subj, this_cond = this_match.group(1), this_match.group(2)

                if dur not in this_cond:
                    continue
                if sync == "sync" and int(this_subj) < 31:
                    continue
                if sync == "async" and int(this_subj) >= 31:
                    continue
                if this_cond not in conds:
                    continue
                if "{}_{}".format(this_subj, this_cond) in excludes:
                    continue

                raw = mne.io.Raw(proc_dir+filename,preload=True)
                picks = mne.pick_types(raw.info, eeg=True)
                bcf = BadChannelFind(picks, thresh=0.5)
                bad_chans = bcf.recommend(raw)
                print(bad_chans)
                raw.info["bads"].extend(bad_chans)

                picks = ["Fz", "FC1", "FC2"]
                psds, freqs = psd_multitaper(raw, fmax=2,
                                             picks=picks, n_jobs=n_jobs)
                psds = psds.mean(axis=0)
                fmax = freqs[np.argmax(psds)]

                raw_data = raw.get_data()
                left_inds = mne.pick_channels(raw.ch_names, left_chans)
                right_inds = mne.pick_channels(raw.ch_names, right_chans)
                left_data = raw_data[left_inds,].mean(axis=0, keepdims=True)
                right_data = raw_data[right_inds,].mean(axis=0, keepdims=True)
                lr_data = np.vstack([left_data, right_data])
                info = mne.create_info(["left", "right"], raw.info["sfreq"], ch_types="eeg")
                lr_raw = mne.io.RawArray(lr_data, info)
                lr_raw = raw.copy().add_channels([lr_raw], force_update_info=True)
                lr_raw.pick_channels(["left", "right"])

                if "5m" in this_cond:
                    seconds = 300
                elif "2m" in this_cond:
                    seconds = 120
                else:
                    seconds = 30

                events, descs = mne.events_from_annotations(lr_raw, regexp="BAD_Stimulation")
                if len(events) > 5:
                    events = events[:5]
                try:
                    epo = mne.Epochs(lr_raw, events, baseline=None, tmin=2, tmax=30+seconds,
                                     reject_by_annotation=False)
                except:
                    print("\n{} could not be segmented.\n".format(this_match))
                    bad_stims.append("{}_{}".format(this_subj,this_cond))
                    continue
                complex = tfr_morlet(epo, [fmax], 3, return_itc=False, n_jobs=4,
                                     average=False, output="complex")
                power = (complex.data * complex.data.conj()).real
                phase = np.angle(complex.data)
                pow = power.mean(axis=1)
                gw_len = np.round(1 * epo.info["sfreq"]).astype(int)
                gauss_win = np.exp(-0.5*((np.arange(gw_len)-gw_len/2)/(0.5*gw_len/2))**2)
                for epo_idx in range(len(epo)):
                    pow[epo_idx, 0,] = np.convolve(pow[epo_idx, 0,], gauss_win, mode="same")
                power_list.append(pow)

                left_idx = mne.pick_channels(epo.ch_names, ["left"])
                right_idx = mne.pick_channels(epo.ch_names, ["right"])

                phase_diff = np.angle(np.exp(0+1j*(phase[:, left_idx,] - phase[:, right_idx,])))[:,0,]
                phase_list.append(phase_diff)
                fmax_list.extend(list(np.ones(len(epo))*fmax))
                subid_list.extend(["{}_{}".format(this_subj,this_cond) for x in range(len(epo))])

                lr_raw.save("{}lr_bad_caf_NAP_{}_{}-raw.fif".format(proc_dir, this_subj, this_cond), overwrite=True)

        phase_array = np.vstack(phase_list)
        power_array = np.vstack(power_list)

        # info = mne.create_info(["frontal"],epo.info["sfreq"],ch_types="misc")
        # grand_epo = mne.EpochsArray(power_array, info)
        # grand_epo.save("{}{}_{}_stim_tfr-epo.fif".format(proc_dir, dur, sync),
        #                overwrite=True)
        # fig = grand_epo.plot_image(picks="frontal")[0]
        # plt.suptitle("{}_{}".format(dur, sync))
        # fig.set_size_inches(38.4, 21.6)
        # plt.savefig("{}_{}_stim_tfr.tif".format(dur, sync))

        # info = mne.create_info(["phase_diff"],epo.info["sfreq"],ch_types="misc")
        # grand_epo = mne.EpochsArray(phase_array, info)
        # grand_epo.save("{}{}_{}_stim_phasediff-epo.fif".format(proc_dir, dur, sync),
        #                overwrite=True)
        # fig = grand_epo.plot_image(picks="phase_diff")[0]
        # plt.suptitle("{}_{}".format(dur, sync))
        # fig.set_size_inches(38.4, 21.6)
        # plt.savefig("{}_{}_stim_phasediff.tif".format(dur, sync))

        pa = phase_array[...,epo.time_as_index(20)[0]:]
        msqrt = np.sqrt((pa**2).mean(-1)).squeeze()
        plt.figure()
        plt.hist(msqrt, bins=50)
        plt.xlim(0,np.pi)
        plt.suptitle("{}_{}".format(dur, sync))
        inds = np.where(msqrt>msqrt_thresh["{}_{}".format(dur,sync)])[0]
        if len(inds):
            bads = list(set([subid_list[x] for x in np.nditer(inds)]))
            bad_stims.extend(bads)
