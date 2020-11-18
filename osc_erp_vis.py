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
time_win = (0.25, 0.4)
freq_win = (12, 17)
ylim = {"misc":(-10,10)}
chan = "central"

epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
epo.apply_baseline((-1.5,-1.25))
df = epo.metadata
sub_inds = df["Subj"].values.astype(int) >= 31
tfr = read_tfrs("{}grand_{}-tfr.h5".format(proc_dir, chan))[0]
epo.crop(tmin=tfr.times[0], tmax=tfr.times[-1])

# plot tfrs
evo_SO = epo["Cond=='{}' or Cond=='{}' or Cond=='sham' and OscType=='SO'".format(conds[0],conds[1])].average()
evo_data = evo_SO.data
evo_data = (evo_data - evo_data.min()) / (evo_data.max() - evo_data.min())
evo_data = evo_data*3 + 12
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

tfr_sham = this_tfr["Cond=='sham' and OscType=='SO'"].average()
tfr_cond0 = this_tfr["Cond=='{}' and OscType=='SO'".format(conds[0])].average()
tfr_cond1 = this_tfr["Cond=='{}' and OscType=='SO'".format(conds[1])].average()
tfr_both = this_tfr["(Cond=='{}' or Cond=='{}') and OscType=='SO'".format(conds[0],conds[1])].average()

fig, axes = plt.subplots(1,3)
tfr_cond0.plot(vmin=vmin, vmax=vmax, axes=axes[0])
axes[0].set_title(conds[0])
tfr_sham.plot(vmin=vmin, vmax=vmax, axes=axes[1])
axes[1].set_title("sham")
(tfr_cond0-tfr_sham).plot(vmin=vmin, vmax=vmax, axes=axes[2])
axes[2].set_title("{}-sham".format(conds[0]))
fig.suptitle("TFR SO, Z-scores from baseline")

fig, axes = plt.subplots(1,3)
tfr_cond1.plot(vmin=vmin, vmax=vmax, axes=axes[0])
axes[0].set_title(conds[1])
tfr_sham.plot(vmin=vmin, vmax=vmax, axes=axes[1])
axes[1].set_title("sham")
(tfr_cond1-tfr_sham).plot(vmin=vmin, vmax=vmax, axes=axes[2])
axes[2].set_title("{}-sham".format(conds[1]))
fig.suptitle("TFR SO, Z-scores from baseline")

fig, axes = plt.subplots(1,3)
tfr_both.plot(vmin=vmin, vmax=vmax, axes=axes[0], colorbar=False)
axes[0].set_title("Average {} and {} stimulation".format(conds[0], conds[1]))
axes[0].plot(tfr.times, evo_data[0,], color="gray", alpha=0.5, linewidth=10)
tfr_sham.plot(vmin=vmin, vmax=vmax, axes=axes[1], colorbar=False)
axes[1].set_title("sham")
axes[1].plot(tfr.times, evo_data[0,], color="gray", alpha=0.5, linewidth=10)
(tfr_both-tfr_sham).plot(vmin=vmin, vmax=vmax, axes=axes[2])
axes[2].set_title("Average {} and {} - sham".format(conds[0], conds[1]))
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

e = epo.copy().filter(l_freq=0.16,h_freq=4.25,n_jobs=4)
e = epo["Cond=='{}' or Cond=='{}' or Cond=='sham'".format(conds[0],conds[1])]

# erp images
evos = []
for osc in oscs:
    this_e = e["OscType=='{}'".format(osc)]
    this_e.plot_image(picks=chan)
    plt.suptitle("{}, {} and sham: {}".format(conds[0], conds[1], osc))
    evos.append(this_e.average())
    evos[-1].comment = osc
mne.viz.plot_compare_evokeds(evos, picks=chan)

# # erps
# colors, styles = [], []
# for col in ["blue", "red", "green", "cyan", "purple"]:
#     for sty in ["dotted", "solid"]:
#         colors.append(col)
#         styles.append(sty)
# for osc in ["SO"]:
#     epo0 = e["OscType=='{}'".format(osc)]
#     for ort in [chan]:
#         for cond in ["eig", "fix"]:
#             for timelen in ["2m"]:
#                 epo1 = epo0["Cond=='{}{}'".format(cond,timelen)]
#                 evo_inds = []
#                 for ind in range(5):
#                     epo2 = epo1["Index=='{}'".format(ind)]
#                     for pp in ["Pre", "Post"]:
#                         epo3 = epo2["PrePost=='{}'".format(pp)]
#                         evo = epo3.average(picks=chan)
#                         evo.comment = "{} {} {}{} Stimulation {} {}".format(ort, osc, cond,
#                                                                 timelen, ind+1, pp)
#                         evo_inds.append(evo)
#
#                 mne.viz.plot_compare_evokeds(evo_inds, picks=chan,
#                                              colors=colors, linestyles=styles,
#                                              ylim=ylim)
#                 plt.legend(loc="lower left")
#
#         epo1 = epo0["Cond=='sham'"]
#         evo_inds = []
#         for ind in range(5):
#             epo2 = epo1["Index=='{}'".format(ind)]
#             for pp in ["Pre", "Post"]:
#                 epo3 = epo2["PrePost=='{}'".format(pp)]
#                 evo = epo3.average(picks=chan)
#                 evo.comment = "{} {} Sham Stimulation {} {}".format(ort, osc, ind+1, pp)
#                 evo_inds.append(evo)
#         mne.viz.plot_compare_evokeds(evo_inds, picks=chan,
#                                      colors=colors, linestyles=styles,
#                                      ylim=ylim)
#         plt.legend(loc="lower left")
