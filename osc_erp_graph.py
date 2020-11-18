import mne
from mne.time_frequency import read_tfrs

import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
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
freq_win = (12, 17)
ylim = {"misc":(-2e-9,2e-9)}
#ylim = None
vmin, vmax = -1e-9, 1e-9

epo = mne.read_epochs("{}grand_central-epo.fif".format(proc_dir), preload=True)
df = epo.metadata
sub_inds = df["Subj"].values.astype(int) >= 31
tfr = read_tfrs("{}grand_central-tfr.h5".format(proc_dir))[0]
epo.crop(tmin=tfr.times[0],tmax=tfr.times[-1])

# plot tfrs
evo_SO = epo["(Cond=='fix30s' or Cond=='eig30s') and OscType=='SO'"].average()
evo_data = evo_SO.data
evo_data = (evo_data - evo_data.min()) / (evo_data.max() - evo_data.min())
evo_data = evo_data*3 + 12
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

tfr_sham = this_tfr["Cond=='sham' and OscType=='SO'"].average()
tfr_stim = this_tfr["Cond!='sham' and OscType=='SO'"].average()

fig, axes = plt.subplots(1,3)
tfr_stim.plot(vmin=vmin, vmax=vmax, axes=axes[0], colorbar=False)
axes[0].set_title("Stimulation")
axes[0].plot(tfr.times, evo_data[0,], color="gray", alpha=0.5, linewidth=10)
tfr_sham.plot(vmin=vmin, vmax=vmax, axes=axes[1], colorbar=False)
axes[1].set_title("Sham")
axes[1].plot(tfr.times, evo_data[0,], color="gray", alpha=0.5, linewidth=10)
(tfr_stim-tfr_sham).plot(vmin=vmin, vmax=vmax, axes=axes[2])
axes[2].set_title("Stimulation-Sham")
axes[2].plot(tfr.times, evo_data[0,], color="gray", alpha=0.5, linewidth=10)
fig.suptitle("TFR SO, Z-scores from baseline")

# add tfr in freq band of interest as channel
freq_inds = (list(tfr.freqs).index(freq_win[0]), list(tfr.freqs).index(freq_win[1]))
time_inds = (list(tfr.times).index(time_win[0]), list(tfr.times).index(time_win[1]))
tfr_chan = tfr.data[:,0,freq_inds[0]:freq_inds[1],:].mean(axis=1)
tfr_chan = np.expand_dims(tfr_chan, 1)
tfr_epo = mne.EpochsArray(tfr_chan, mne.create_info(["TFR"],tfr.info["sfreq"],
                          ch_types="misc"))
epo = epo.add_channels([tfr_epo], force_update_info=True)

e = epo.copy().filter(l_freq=0.3,h_freq=3,n_jobs=4)
#e = epo["Cond=='{}' or Cond=='{}' or Cond=='sham'".format(conds[0],conds[1])]

# erp images
evos = []
for osc in oscs:
    this_e = e["OscType=='{}'".format(osc)]
    this_e.plot_image(picks=pick)
    plt.suptitle("30 second stimulations and Sham, {}".format(osc))
    evos.append(this_e.average(picks=pick))
    evos[-1].comment = osc
mne.viz.plot_compare_evokeds(evos,picks=pick)

# # erps by prepost
# colors, styles = [], []
# for col in ["blue", "red", "green", "cyan", "purple"]:
#     for sty in ["dotted", "solid"]:
#         colors.append(col)
#         styles.append(sty)
# for osc in ["SO"]:
#     epo0 = e["OscType=='{}'".format(osc)]
#     for cond in ["eig", "fix"]:
#         for timelen in ["30s"]:
#             epo2 = epo0["Cond=='{}{}'".format(cond,timelen)]
#             evo_inds = []
#             for ind in range(5):
#                 epo3 = epo2["Index=='{}'".format(ind)]
#                 for pp in ["Pre", "Post"]:
#                     epo4 = epo3["PrePost=='{}'".format(pp)]
#                     evo = epo4.average(picks=pick)
#                     evo.comment = "{} {}{} Stimulation {} {}".format(osc, cond,
#                                                             timelen, ind+1, pp)
#                     evo_inds.append(evo)
#
#             mne.viz.plot_compare_evokeds(evo_inds, picks=pick,
#                                          colors=colors, linestyles=styles,
#                                          ylim=ylim)
#             plt.legend(loc="lower left")
#
#     epo2 = epo0["Cond=='sham'"]
#     evo_inds = []
#     for ind in range(5):
#         epo3 = epo2["Index=='{}'".format(ind)]
#         for pp in ["Pre", "Post"]:
#             epo4 = epo3["PrePost=='{}'".format(pp)]
#             evo = epo4.average(picks=pick)
#             evo.comment = "{} Sham Stimulation {} {}".format(osc, ind+1, pp)
#             evo_inds.append(evo)
#     mne.viz.plot_compare_evokeds(evo_inds, picks=pick,
#                                  colors=colors, linestyles=styles,
#                                  ylim=ylim)
#     plt.legend(loc="lower left")

# erps by condition
#sub_inds = epo.metadata["Subj"].values.astype(int) >= 31
#epo = epo[sub_inds]
colors, styles = [], []
for col in ["blue", "red", "green"]:
    for sty in ["dotted", "solid"]:
        colors.append(col)
        styles.append(sty)
colors.append("cyan")
styles.append("solid")
for osc in ["SO"]:
    epo0 = epo["OscType=='{}'".format(osc)]
    evo_inds = []
    for timelen in ["30s", "2m","5m"]:
        for cond in ["eig", "fix"]:
            epo2 = epo0["Cond=='{}{}'".format(cond,timelen)]
            evo = epo2.average(picks=pick)
            evo.comment = "{} {}{} Stimulation".format(osc, cond, timelen)
            evo_inds.append(evo)

    epo2 = epo0["Cond=='sham'"]
    evo = epo2.average(picks=pick)
    evo.comment = "{} {} Stimulation".format(osc, "sham")
    evo_inds.append(evo)
    mne.viz.plot_compare_evokeds(evo_inds, picks=pick, ylim=ylim, title="TFR 12-17Hz",
                                 colors=colors, linestyles=styles)
    plt.legend(loc="lower left")

# erps by pureindex
conds = ["sham", "30s", "2m", "5m"]
conds = ["sham", "stim"]
pureinds = [str(idx) for idx in [0,1,3,5,7,9]]
colors = ["black", "blue", "red", "green", "cyan", "magenta"]
styles = ["dotted", "solid", "solid", "solid", "solid", "solid"]
evonames = ["Pre-Stimulation", "Post-Stimulation 1", "Post-Stimulation 2",
            "Post-Stimulation 3", "Post-Stimulation 4", "Post-Stimulation 5"]
for cond in conds:
    if cond == "sham":
        epo0 = epo["Cond=='sham'"]
    elif cond == "stim":
        epo0 = epo["Cond!='sham'"]
    else:
        epo0 = epo["Cond=='fix{}' or Cond=='eig{}'".format(cond, cond)]
    for osc in ["SO"]:
        epo0 = epo0["OscType=='{}'".format(osc)]
        evo_inds = []
        for pureind, evoname in zip(pureinds, evonames):
            epo2 = epo0["PureIndex=='{}'".format(pureind)]
            evo = epo2.average(picks=pick)
            evo.comment = "{} {}".format(osc, evoname)
            evo_inds.append(evo)
        mne.viz.plot_compare_evokeds(evo_inds, picks=pick, ylim=ylim,
                                     title="{} TFR 12-17Hz".format(cond),
                                     colors=colors,
                                     linestyles=styles)
        plt.legend(loc="lower left")
