import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pickle
from os.path import isdir
import mne
from mne.time_frequency import read_tfrs
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

perm_chunk = 64
perm_chunk_n = 64

dur = "30s"
model= "cond"
baseline = "logratio"
chan = "central"
col = "Cond"
osc = "SO"
sync = "sync"
res_dir = proc_dir + "perms/{}/{}/{}/{}/".format(baseline, osc, dur, sync)

cond_keys = {"Intercept":"Intercept",
             "C(Cond, Treatment('sham{}'))[T.eig{}]".format(dur,dur):"eig{}".format(dur),
             "C(Cond, Treatment('sham{}'))[T.fix{}]".format(dur,dur):"fix{}".format(dur),
             "C(Cond, Treatment('sham'))[T.eig{}]".format(dur):"eig{}".format(dur),
             "C(Cond, Treatment('sham'))[T.fix{}]".format(dur):"fix{}".format(dur),
             "C(Stim, Treatment('sham'))[T.stim]":"stim"}

with open("{}main_result_{}.pickle".format(res_dir, model), "rb") as f:
    main_result = pickle.load(f)
stat_conds = list(main_result["raw_t"].keys())
main_tfce_pos = {cond_keys[k]:main_result["tfce_pos"][k] for k in stat_conds}
main_tfce_neg = {cond_keys[k]:main_result["tfce_neg"][k] for k in stat_conds}

perms = []
for pcn in range(perm_chunk_n):
    try:
        with open("{}perm_result_{}_{}_{}.pickle".format(res_dir, perm_chunk, pcn, model), "rb") as f:
            perms.append(pickle.load(f))
    except:
        continue

perm_tfce_pos = {cond_keys[k]:[] for k in stat_conds}
perm_tfce_neg = {cond_keys[k]:[] for k in stat_conds}
for perm in perms:
    for pn in perm:
        #stat_conds = list(main_result["raw_t"].keys())
        for sc in stat_conds:
            perm_tfce_pos[cond_keys[sc]].append(pn["tfce_pos"][sc])
            perm_tfce_neg[cond_keys[sc]].append(pn["tfce_neg"][sc])
perm_tfce_pos = {k:np.array(v).ravel() for k,v in perm_tfce_pos.items()}
perm_tfce_neg = {k:np.array(v).ravel() for k,v in perm_tfce_neg.items()}

pos_thresh = {k:np.percentile(perm_tfce_pos[k],95) for k in perm_tfce_pos.keys()}
neg_thresh = {k:np.percentile(perm_tfce_neg[k],95) for k in perm_tfce_neg.keys()}

epo = mne.read_epochs("{}grand_{}-epo.fif".format(proc_dir, chan), preload=True)
epo.apply_baseline((-2.15,-1.68))
epo.crop(tmin=-1.5, tmax=1.5)
tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr = tfr["Cond=='eig{}' or Cond=='fix{}' or Cond=='sham{}'".format(dur,dur,dur)]
epo = epo["Cond=='eig{}' or Cond=='fix{}' or Cond=='sham{}'".format(dur,dur,dur)]
subjs = np.unique(tfr.metadata["Subj"].values)
# check for missing conditions in each subject
bad_subjs = []
if sync == "sync":
    for subj in list(subjs):
        if int(subj) < 31:
            bad_subjs.append(subj)
for subj in subjs:
    this_df = tfr.metadata.query("Subj=='{}'".format(subj))
    these_conds = list(np.unique(this_df[col].values))
    checks = [c in these_conds for c in list(np.unique(tfr.metadata[col].values))]
    if not all(checks):
        bad_subjs.append(subj)
bad_subjs = list(set(bad_subjs))
for bs in bad_subjs:
    print("Removing subject {}".format(bs))
    tfr = tfr["Subj!='{}'".format(bs)]
    epo = epo["Subj!='{}'".format(bs)]

vmin, vmax = -0.2, 0.2
vmin, vmax = -2e-11, 1.1e-10
vmin, vmax = 0, 125
vmin, vmax = -.3, 0
fig, axes = plt.subplots(4, 2, figsize=(38.4, 21.6))
evos = {}
tfrs = {}
evos["sham"+dur] = epo["Cond=='sham{}'".format(dur)].average()
evos["sham"+dur].comment = "sham"
#tfrs["sham"+dur] = tfr["Cond=='sham{}'".format(dur)].average()
tfrs["sham"+dur] = read_tfrs("{}predicted_cond_{}_{}_{}_{}_sham-tfr.h5".format(proc_dir, osc, baseline, dur, sync))[0]
tfrs["sham"+dur].plot(picks="central", axes=axes[1][0], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis")
tfrs["sham"+dur].plot(picks="central", axes=axes[1][1], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis")
for cond_idx, cond in enumerate(["fix"+dur, "eig"+dur]):
    evos[cond] = epo["Cond=='{}'".format(cond)].average()
    evos[cond].comment = cond
    mne.viz.plot_compare_evokeds([evos["sham"+dur],evos[cond]], picks="central",
                                 axes=axes[0][cond_idx], ylim={"eeg":(-30,15)},
                                 styles={"sham":{"linewidth":4},cond:{"linewidth":5}},
                                 title="Slow Oscillations: {}".format(cond))
    tfrs[cond] = read_tfrs("{}predicted_cond_{}_{}_{}_{}_{}-tfr.h5".format(proc_dir, osc, baseline, dur, sync, cond))[0]
    tfrs[cond].plot(picks="central", axes=axes[2][cond_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis")
    mask = (main_tfce_pos[cond]>pos_thresh[cond]).T
    contrast = tfrs[cond]-tfrs["sham"+dur]
    #contrast.data[0,] = main_tfce_pos[cond].T
    contrast.plot(picks="central", axes=axes[3][cond_idx],
                  colorbar=False, vmin=vmin, vmax=-vmin, cmap="RdBu_r",
                  mask=mask, mask_style="contour", mask_alpha=0.6,
                  mask_cmap="RdBu_r")

axes[1][0].set_title("sham TFR")
axes[1][1].set_title("sham TFR")
axes[2][0].set_title("fix{} TFR".format(dur))
axes[2][1].set_title("eig{} TFR".format(dur))
axes[3][0].set_title("fix{}-sham TFR".format(dur))
axes[3][1].set_title("eig{}-sham TFR".format(dur))
plt.suptitle("{}, {}".format(dur, sync))
plt.tight_layout()

plt.savefig("{}_{}_{}_{}.tif".format(baseline, osc, dur, sync))
