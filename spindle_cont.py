import mne
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

n_jobs = 8
spindle_freq = np.arange(10,21)
chans = ["central"]
osc_types = ["SO", "deltO"]
sfreq = 50.
thresh = 99.9
epo_pref = ""
zboot_n = None
bl = (-2.35,-1.5)
crop = (-1.5,1.5)

for chan in chans:
    epo = mne.read_epochs("{}{}grand_{}_finfo-epo.fif".format(proc_dir, epo_pref,chan),
                          preload=True)
    epo.resample(sfreq, n_jobs="cuda")
    power = tfr_morlet(epo, spindle_freq, n_cycles=5, average=False,
                       return_itc=False, n_jobs=n_jobs)
    power.crop(tmin=-2.35, tmax=2.35) # get rid of edge effects

    power_mean = power.copy().apply_baseline((bl[0],bl[1]), mode="mean")
    power_mean.crop(tmin=crop[0],tmax=crop[1])
    power_mean.save("{}{}grand_{}_mean-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_log = power.copy()
    power_log.data = np.log10(power_log.data)
    power_log = power_log.apply_baseline((bl[0],bl[1]), mode="mean")
    power_log.crop(tmin=crop[0],tmax=crop[1])
    power_log.save("{}{}grand_{}_logmean-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_z = power.copy().apply_baseline((bl[0],bl[1]), mode="zscore")
    power_z.crop(tmin=crop[0],tmax=crop[1])
    power_z.save("{}{}grand_{}_zscore-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_logratio = power.copy().apply_baseline((bl[0],bl[1]), mode="logratio")
    power_logratio.crop(tmin=crop[0],tmax=crop[1])
    power_logratio.save("{}{}grand_{}_logratio-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_zlogratio = power.copy().apply_baseline((bl[0],bl[1]), mode="zlogratio")
    power_zlogratio.crop(tmin=crop[0],tmax=crop[1])
    power_zlogratio.save("{}{}grand_{}_zlogratio-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    power_none = power.copy().crop(tmin=crop[0],tmax=crop[1])
    power_none.save("{}{}grand_{}_none-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)

    if zboot_n:
        df = power.metadata.copy()
        power_zboot = power.copy()
        zboot_data = power_zboot.data.copy()
        subjs = list(df["Subj"].unique())
        for subj in subjs:
            subj_power_zboot = power_zboot["Subj=='{}'".format(subj)]
            conds = list(subj_power_zboot.metadata["Cond"].unique())
            for cond in conds:
                this_power_zboot = subj_power_zboot["Cond=='{}'".format(cond)]
                this_power_zboot.crop(tmin=bl[0], tmax=bl[1])
                data = this_power_zboot.data[:,0,] # assumes one electrode
                trial_n = data.shape[0]
                freq_n = data.shape[1]
                # cycle through frequencies; assumes 2nd index (after removing channel index)
                perm_results = np.zeros((zboot_n, freq_n))
                for f_idx in range(freq_n):
                    f_data = data[:,f_idx,].flatten()
                    for z_idx in range(zboot_n):
                        np.random.shuffle(f_data)
                        perm_results[z_idx, f_idx] = f_data[:trial_n].mean()

                sur_mu = perm_results.mean(axis=0)
                sur_std = perm_results.std(axis=0)

                inds = np.where((df["Subj"]==subj) & (df["Cond"]==cond))[0]
                new_data = zboot_data[inds,].copy()
                for f_idx in range(freq_n):
                    new_data[:,:,f_idx,] = (new_data[:,:,f_idx,] - sur_mu[f_idx]) / sur_std[f_idx]
                zboot_data[inds,] = new_data
        power_zboot.data = zboot_data
        power_zboot.crop(tmin=crop[0],tmax=crop[1])
        power_zboot.save("{}{}grand_{}_zboot-tfr.h5".format(proc_dir, epo_pref, chan), overwrite=True)
