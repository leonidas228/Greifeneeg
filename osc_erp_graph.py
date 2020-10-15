import mne
from mne.time_frequency import read_tfrs

import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 16}
matplotlib.rc('font', **font)

import numpy as np
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

oscs = ["SO", "deltO"]
conds = ["eig30s", "fix30s"]
pick = "TFR"
time_win = (0.25, 0.4)
freq_win = (14, 16)
ylim = {"misc":(-10,10)}

epo = mne.read_epochs("{}grand-epo.fif".format(proc_dir), preload=True)
df = epo.metadata
sub_inds = df["Subj"].values.astype(int) >= 31
tfr = read_tfrs("{}grand-tfr.h5".format(proc_dir))[0]

# plot tfrs
vmin, vmax = -4, 4
this_tfr = tfr["Cond=='{}' or Cond=='{}' or Cond=='sham'".format(conds[0],conds[1])]
tfr_SO = this_tfr["OscType=='SO'"].average()
tfr_deltO = this_tfr["OscType=='deltO'"].average()

fig, axes = plt.subplots(1,3)
tfr_SO.plot(vmin=vmin, vmax=vmax, axes=axes[0])
axes[0].set_title("SO")
tfr_deltO.plot(vmin=vmin, vmax=vmax, axes=axes[1])
axes[1].set_title("deltO")
(tfr_SO-tfr_deltO).plot(vmin=vmin, vmax=vmax, axes=axes[2])
axes[2].set_title("SO-deltO")
fig.suptitle("TFR, Z-scores from baseline")

# add tfr in freq band of interest as channel
freq_inds = (list(tfr.freqs).index(freq_win[0]), list(tfr.freqs).index(freq_win[1]))
time_inds = (list(tfr.times).index(time_win[0]), list(tfr.times).index(time_win[1]))
tfr_chan = tfr.data[:,0,freq_inds[0]:freq_inds[1],:].mean(axis=1)
tfr_chan = np.expand_dims(tfr_chan, 1)
tfr_epo = mne.EpochsArray(tfr_chan, mne.create_info(["TFR"],tfr.info["sfreq"],
                          ch_types="misc"))
epo = epo.add_channels([tfr_epo], force_update_info=True)

#e = epo.copy().filter(l_freq=0.3,h_freq=3,n_jobs=4)
e = epo["Cond=='{}' or Cond=='{}' or Cond=='sham'".format(conds[0],conds[1])]

# # erp images
# evos = []
# for osc in oscs:
#     this_e = e["OscType=='{}'".format(osc)]
#     this_e.plot_image(picks=pick)
#     plt.suptitle("30 second stimulations and Sham, {}".format(osc))
#     evos.append(this_e.average())
#     evos[-1].comment = osc
# mne.viz.plot_compare_evokeds(evos,picks=pick)

colors, styles = [], []
for col in ["blue", "red", "green", "cyan", "purple"]:
    for sty in ["dotted", "solid"]:
        colors.append(col)
        styles.append(sty)

# erps
for osc in ["SO"]:
    epo0 = e["OscType=='{}'".format(osc)]
    for ort in ["central"]:
        epo1 = epo0["Ort=='{}'".format(ort)]
        for cond in ["eig", "fix"]:
            for timelen in ["30s"]:
                epo2 = epo1["Cond=='{}{}'".format(cond,timelen)]
                evo_inds = []
                for ind in range(5):
                    epo3 = epo2["Index=='{}'".format(ind)]
                    for pp in ["Pre", "Post"]:
                        epo4 = epo3["PrePost=='{}'".format(pp)]
                        evo = epo4.average(picks=pick)
                        evo.comment = "{} {} {}{} Stimulation {} {}".format(ort, osc, cond,
                                                                timelen, ind+1, pp)
                        evo_inds.append(evo)

                mne.viz.plot_compare_evokeds(evo_inds, picks=pick,
                                             colors=colors, linestyles=styles,
                                             ylim=ylim)
                plt.legend(loc="lower left")

        epo2 = epo1["Cond=='sham'"]
        evo_inds = []
        for ind in range(5):
            epo3 = epo2["Index=='{}'".format(ind)]
            for pp in ["Pre", "Post"]:
                epo4 = epo3["PrePost=='{}'".format(pp)]
                evo = epo4.average(picks=pick)
                evo.comment = "{} {} Sham Stimulation {} {}".format(ort, osc, ind+1, pp)
                evo_inds.append(evo)
        mne.viz.plot_compare_evokeds(evo_inds, picks=pick,
                                     colors=colors, linestyles=styles,
                                     ylim=ylim)
        plt.legend(loc="lower left")
