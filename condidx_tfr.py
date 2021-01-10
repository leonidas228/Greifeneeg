import mne
import numpy as np
from mne.time_frequency import read_tfrs
from os.path import isdir
import pickle
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

syncs = ["async","sync"]
dur = "30s"
conds = ["sham","fix","eig"]
oscs = ["SO","deltO"]
baseline = "zscore"
to_plot = ["coef", "tval"]
to_plot = ["coef"]

cond_keys = {'Intercept':"sham",
             "C(Cond, Treatment('sham{}'))[T.eig{}]".format(dur,dur):"eig{}".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]".format(dur,dur):"fix{}".format(dur),
             "C(Index, Treatment('Pre'))[T.0]":"sham{}_0".format(dur),
             "C(Index, Treatment('Pre'))[T.1]":"sham{}_1".format(dur),
             "C(Index, Treatment('Pre'))[T.2]":"sham{}_2".format(dur),
             "C(Index, Treatment('Pre'))[T.3]":"sham{}_3".format(dur),
             "C(Index, Treatment('Pre'))[T.4]":"sham{}_4".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.0]".format(dur,dur):"eig{}_0".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.0]".format(dur,dur):"fix{}_0".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.1]".format(dur,dur):"eig{}_1".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.1]".format(dur,dur):"fix{}_1".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.2]".format(dur,dur):"eig{}_2".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.2]".format(dur,dur):"fix{}_2".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.3]".format(dur,dur):"eig{}_3".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.3]".format(dur,dur):"fix{}_3".format(dur),
             "C(Cond, Treatment('sham{}'))[T.eig{}]:C(Index, Treatment('Pre'))[T.4]".format(dur,dur):"eig{}_4".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]:C(Index, Treatment('Pre'))[T.4]".format(dur,dur):"fix{}_4".format(dur)}
key_conds = {v:k for k,v in cond_keys.items()}

inds = ["0","1","2","3","4"]
conds = ["sham","fix","eig"]
title_keys = {"sham":"sham", "fix":"fixed frequency stimulation", "eig":"Eigenfrequency stimulation"}

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()

epo = mne.read_epochs(proc_dir+"grand_central-epo.fif")
epo.resample(tfr.info["sfreq"], n_jobs="cuda")
epo.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
# calculate global ERP min and max for scaling later on
evo = epo.average()
ev_min, ev_max = evo.data.min(), evo.data.max()

for osc in oscs:
    for sync in syncs:
        e = epo.copy()
        # if sync then remove all subjects recorded under asynchronous conditions (<31)
        if sync == "sync":
            subjs = epo.metadata["Subj"]
            bad_subjs = []
            if sync == "sync":
                for subj in list(subjs):
                    if int(subj) < 31:
                        bad_subjs.append(subj)
            bad_subjs = list(set(bad_subjs))
            for bs in bad_subjs:
                print("Removing subject {}".format(bs))
                e = e["Subj!='{}'".format(bs)]

        fig, axes = plt.subplots(5,len(conds)*len(to_plot), figsize=(38.4, 21.6))
        tfr_c = tfr_avg.copy()
        dat_shape = tfr_c.data.shape[1:]
        with open("{}main_fits_{}_condidx_{}_{}_{}.pickle".format(proc_dir,baseline,osc,dur,sync), "rb") as f:
            fits = pickle.load(f)
        modfit = fits["fits"]
        exog_names = fits["exog_names"]

        for ind_idx, ind in enumerate(inds):
            for cond_idx, cond in enumerate(conds):

                # get osc ERP and normalise
                evo = e["OscType=='{}' and Cond=='{}{}' and Index=='{}'".format(osc, cond, dur, ind)].average()
                evo_data = evo.data
                evo_data = (evo_data - ev_min) / (ev_max - ev_min)
                evo_data = evo_data*3 + 12

                data = np.zeros((3, len(modfit)))
                for mf_idx, mf in enumerate(modfit):
                    data[0, mf_idx] = mf.params[exog_names.index(key_conds["{}{}_{}".format(cond,dur,ind)])]
                    data[1, mf_idx] = mf.tvalues[exog_names.index(key_conds["{}{}_{}".format(cond,dur,ind)])]
                    data[2, mf_idx] = mf.pvalues[exog_names.index(key_conds["{}{}_{}".format(cond,dur,ind)])]
                pvals = data[2,].reshape(*dat_shape, order="F")
                pvals[np.isnan(pvals)] = 1
                mask = pvals<0.05
                for val_idx,val in enumerate(to_plot):
                    dat = data[val_idx,].reshape(*dat_shape, order="F")
                    dat[np.isnan(dat)] = 0
                    tfr_c.data[0,] = dat
                    if val == "tval":
                        vmin, vmax = -4, 4
                    else:
                        vmin, vmax = -4, 4
                    tfr_c.plot(picks="central",
                               axes=axes[ind_idx][cond_idx*len(to_plot)+val_idx],
                               colorbar=False, vmin=vmin, vmax=vmax,
                               cmap="viridis", mask=mask, mask_style="contour")
                    axes[ind_idx][cond_idx*len(to_plot)+val_idx].plot(tfr.times,
                                                                      evo_data[0,],
                                                                      color="gray",
                                                                      alpha=0.8,
                                                                      linewidth=10)
                    if cond_idx*len(to_plot)+val_idx == len(conds)*len(to_plot)-1:
                        rax = axes[ind_idx][cond_idx*len(to_plot)+val_idx].twinx()
                        plt.ylabel("Stimulation {}".format(int(ind)+1))
                        plt.yticks(ticks=[], labels=[])
                    if ind_idx == 0:
                        axes[ind_idx][cond_idx*len(to_plot)+val_idx].set_title(title_keys[cond])

        if sync == "sync":
            plt.suptitle("{}, {} stimulation, synchronised only".format(osc, dur))
        else:
            plt.suptitle("{}, {} stimulation, non-synchronised included".format(osc, dur))
        plt.tight_layout()
        plt.savefig("condidx/{}_{}_{}_{}_byindex.tif".format(baseline, osc, dur, sync))

        # fig, axes = plt.subplots(2,2, figsize=(38.4, 21.6))
        # for cond_idx,cond in enumerate(["fix","eig"]):
        #     data = np.zeros((3, len(modfit)))
        #     for mf_idx, mf in enumerate(modfit):
        #         data[0, mf_idx] = mf.params[exog_names.index(key_conds["{}{}".format(cond,dur)])]
        #         data[1, mf_idx] = mf.tvalues[exog_names.index(key_conds["{}{}".format(cond,dur)])]
        #         data[2, mf_idx] = mf.pvalues[exog_names.index(key_conds["{}{}".format(cond,dur)])]
        #     pvals = data[2,].reshape(*dat_shape, order="F")
        #     pvals[np.isnan(pvals)] = 1
        #     mask = pvals < 0.05
        #     for val_idx,val in enumerate(["coef","tval"]):
        #         dat = data[val_idx,].reshape(*dat_shape, order="F")
        #         dat[np.isnan(dat)] = 0
        #         tfr_c.data[0,] = dat
        #         if val == "tval":
        #             vmin, vmax = -4, 4
        #         else:
        #             vmin, vmax = -4, 4
        #         tfr_c.plot(picks="central", axes=axes[cond_idx][val_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis", mask=mask, mask_style="contour")
        # plt.savefig("condidx/{}_{}_{}_{}_main.tif".format(baseline, osc, dur, sync))
