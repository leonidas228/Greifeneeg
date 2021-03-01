import mne
from mne.time_frequency import read_tfrs
import argparse
import pandas as pd
from os.path import isdir
import re
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pickle
import seaborn as sns
import matplotlib
font = {'weight' : 'bold',
        'size'   : 26}
matplotlib.rc('font', **font)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"
img_dir = root_dir+"images/"

eig_freqs = {'002': 0.8135739656, '003': 0.8388119505, '005': 0.6697439105,
             '006': 0.7832908796, '007': 0.7456094526, '009': 0.7788409575,
             '013': 0.6714744633, '015': 0.8900342154, '016': 0.8195879177,
             '017': 0.807011053, '018': 0.8468273959, '021': 0.5211960758,
             '022': 0.5030628337, '024': 0.5370360281, '025': 0.5080682169,
             '026': 0.511158806, '027': 0.5085371744, '028': 0.6724768668,
             '031': 0.8641548566, '033': 0.9979009669, '035': 0.6158327557,
             '037': 0.6919189534, '038': 0.6330339807, '043': 0.6852392211,
             '044': 0.6559392045, '045': 0.6264749245, '046': 0.6615336116,
             '047': 0.8233948546, '048': 0.7762510701, '050': 0.6685955766,
             '051': 0.7523264618, '053': 0.8741048686, '054': 0.6611531633}

bad_list = {}
bad_list["under"] = [k for k,v in eig_freqs.items() if v > .75]
bad_list["over"] = [k for k,v in eig_freqs.items() if v <= .75]

n_jobs = 8
chan = "central"
baseline = "zscore"
osc = "SO"
durs = ["30s","2m","5m"]
conds = ["sham","eig","fix"]
vmin, vmax = -2.5, 2.5

mask = None

for overunder in ["under", "over"]:
    tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
    tfr = tfr["OscType=='{}' and PrePost=='Post'".format(osc)]
    epo = mne.read_epochs(proc_dir+"grand_central_finfo-epo.fif")
    epo = epo["OscType=='{}' and PrePost=='Post'".format(osc)]
    epo.resample(tfr.info["sfreq"], n_jobs="cuda")
    epo.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
    # calculate global ERP min and max for scaling later on
    evo = epo.average()
    ev_min, ev_max = evo.data.min(), evo.data.max()
    avg_fig, avg_axes = plt.subplots(3, 3, figsize=(38.4, 21.6))
    mods = []
    for dur_idx,dur in enumerate(durs):
        subjs = np.unique(tfr.metadata["Subj"].values)
        col = "Cond"
        bad_subjs = []
        # remove subjects by list
        for subj in list(subjs):
            if subj in bad_list[overunder]:
                bad_subjs.append(subj)
        # remove all subjects with missing conditions or not meeting synchronicity criterion
        bad_subjs = list(set(bad_subjs))
        for bs in bad_subjs:
            print("Removing subject {}".format(bs))
            tfr = tfr["Subj!='{}'".format(bs)]
            epo = epo["Subj!='{}'".format(bs)]

        subjs = np.unique(tfr.metadata["Subj"].values)
        for cond_idx, cond in enumerate(conds):

            # get osc ERP and normalise
            evo = epo["Cond=='{}{}'".format(cond,dur)].average()
            evo_data = evo.data
            evo_data = (evo_data - ev_min) / (ev_max - ev_min)
            evo_data = evo_data*3 + 13

            this_tfr = tfr["Cond=='{}{}'".format(cond,dur)]
            this_avg = this_tfr.average()
            this_avg.plot(picks="central", axes=avg_axes[dur_idx][cond_idx],
                          colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis",
                          mask=mask, mask_style="contour")
            avg_axes[dur_idx][cond_idx].plot(tfr.times, evo_data[0,],
                                             color="gray", alpha=0.8,
                                             linewidth=10)
            if dur_idx == 0:
                avg_axes[dur_idx][cond_idx].set_title("{}".format(cond))
            if cond_idx == len(conds)-1:
                avg_rax = avg_axes[dur_idx][cond_idx].twinx()
                avg_rax.set_ylabel("{}".format(dur))
                avg_rax.set_yticks([])

    avg_fig.suptitle("Raw average {}, Eigenfrequency {} 0.75Hz".format(osc, overunder))
    avg_fig.savefig("../images/{}_{}_rawavg.tif".format(osc, overunder))
