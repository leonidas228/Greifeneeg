import mne
from mne.stats.cluster_level import _find_clusters
import numpy as np
from mne.time_frequency import read_tfrs
from os.path import isdir
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

def tfce_correct(data, tfce_thresh):
    pos_data = data.copy()
    pos_data[pos_data<0] = 0
    neg_data = data.copy()
    neg_data[neg_data>0] = 0
    pos_clusts = _find_clusters(pos_data, tfce_thresh)[1].reshape(data.shape)
    neg_clusts = _find_clusters(neg_data, tfce_thresh)[1].reshape(data.shape)
    out_data = np.zeros_like(data) + pos_clusts - neg_clusts
    return out_data

def cond2vec(exog_names, params, keys_cond):
    out_vec = np.zeros(len(exog_names))
    for param in params:
        out_vec[exog_names.index(keys_cond[param])] = 1
    return out_vec

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

cond_keys = {"Intercept":"Sham 30s",
             "C(StimType, Treatment('sham'))[T.eig]":"Eigenfrequency",
             "C(StimType, Treatment('sham'))[T.fix]":"Fixed frequency",
             "C(Dur, Treatment('30s'))[T.2m]":"2m",
             "C(Dur, Treatment('30s'))[T.5m]":"5m",
             "C(StimType, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.2m]":"Eigenfrequency 2m",
             "C(StimType, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.2m]":"Fixed frequency 2m",
             "C(StimType, Treatment('sham'))[T.eig]:C(Dur, Treatment('30s'))[T.5m]":"Eigenfrequency 5m",
             "C(StimType, Treatment('sham'))[T.fix]:C(Dur, Treatment('30s'))[T.5m]":"Fixed frequency 5m"
            }

cond_exogs =   {"Sham 30s":["Intercept (Sham 30s)"],
                "Eigenfrequency 30s":["Intercept (Sham 30s)", "Eigenfrequency"],
                "Fixed frequency 30s":["Intercept (Sham 30s)", "Fixed frequency"],
                "Sham 2m":["Intercept (Sham 30s)", "2m"],
                "Eigenfrequency 2m":["Intercept (Sham 30s)", "2m", "Eigenfrequency", "Eigenfrequency 2m"],
                "Fixed frequency 2m":["Intercept (Sham 30s)", "2m", "Fixed frequency", "Fixed frequency 2m"],
                "Sham 5m":["Intercept (Sham 30s)", "5m"],
                "Eigenfrequency 5m":["Intercept (Sham 30s)", "5m", "Eigenfrequency", "Eigenfrequency 5m"],
                "Fixed frequency 5m":["Intercept (Sham 30s)", "5m", "Fixed frequency", "Fixed frequency 5m"]}

cond_exogs_syncfact =   {"Sham 30s synchronised":["Intercept (sham30s synchronised)"],
                "Eigenfrequency 30s synchronised":["Intercept (sham30s synchronised)", "Eigenfrequency"],
                "Fixed frequency 30s synchronised":["Intercept (sham30s synchronised)", "Fixed frequency"],
                "Sham 2m synchronised":["Intercept (sham30s synchronised)", "2m"],
                "Eigenfrequency 2m synchronised":["Intercept (sham30s synchronised)", "2m", "Eigenfrequency", "Eigenfrequency 2m"],
                "Fixed frequency 2m synchronised":["Intercept (sham30s synchronised)", "2m", "Fixed frequency", "Fixed frequency 2m"],
                "Sham 5m synchronised":["Intercept (sham30s synchronised)", "5m"],
                "Eigenfrequency 5m synchronised":["Intercept (sham30s synchronised)", "5m", "Eigenfrequency", "Eigenfrequency 5m"],
                "Fixed frequency 5m synchronised":["Intercept (sham30s synchronised)", "5m", "Fixed frequency", "Fixed frequency 5m"],

                "Sham 30s non-synchronised":["Intercept (sham30s synchronised)", "non-synchronised"],
                "Eigenfrequency 30s non-synchronised":["Intercept (sham30s synchronised)", "Eigenfrequency", "non-synchronised", "Eigenfrequency non-synchronised"],
                "Fixed frequency 30s non-synchronised":["Intercept (sham30s synchronised)", "Fixed frequency", "non-synchronised", "Fixed frequency non-synchronised"],
                "Sham 2m non-synchronised":["Intercept (sham30s synchronised)", "2m", "non-synchronised", "2m non-synchronised"],
                "Eigenfrequency 2m non-synchronised":["Intercept (sham30s synchronised)", "2m", "Eigenfrequency", "non-synchronised", "2m non-synchronised", "Eigenfrequency 2m", "Eigenfrequency non-synchronised", "Eigenfrequency 2m non-synchronised"],
                "Fixed frequency 2m non-synchronised":["Intercept (sham30s synchronised)", "2m", "Fixed frequency", "non-synchronised", "2m non-synchronised", "Fixed frequency 2m", "Fixed frequency non-synchronised", "Fixed frequency 2m non-synchronised"],
                "Sham 5m non-synchronised":["Intercept (sham30s synchronised)", "5m", "non-synchronised", "5m non-synchronised"],
                "Eigenfrequency 5m non-synchronised":["Intercept (sham30s synchronised)", "5m", "Eigenfrequency", "non-synchronised", "5m non-synchronised", "Eigenfrequency 5m", "Eigenfrequency non-synchronised", "Eigenfrequency 5m non-synchronised"],
                "Fixed frequency 5m non-synchronised":["Intercept (sham30s synchronised)", "5m", "Fixed frequency", "non-synchronised", "5m non-synchronised", "Fixed frequency 5m", "Fixed frequency non-synchronised", "Fixed frequency 5m non-synchronised"],
                }

durs = ["30s", "2m", "5m"]
conds = ["sham","fix","eig"]
osc = "SO"
baseline = "zscore"
sync_fact = "rsyncfact"
use_group = "group"
badsubjs = "no2,3,28"
if baseline == "zscore":
    vmin, vmax = -2.5, 2.5
elif baseline == "logmean":
    vmin, vmax = -.35, .35
tfce_thresh = dict(start=0, step=0.2)
perm_thresh = .05
cmap = "seismic"
interp = True

if sync_fact == "syncfact":
    # adjust the keys
    new_cond_keys = {k+":C(Sync, Treatment('sync'))[T.async]":v+" non-synchronised"
                 for k,v in cond_keys.items() if "Intercept" not in k}
    cond_keys["C(Sync, Treatment('sync'))[T.async]"] = "non-synchronised"
    cond_keys = {**cond_keys, **new_cond_keys}
    cond_exogs = cond_exogs_syncfact

cond_keys["Intercept"] = "Intercept (Sham 30s"
if sync_fact == "syncfact":
    cond_keys["Intercept"] += " synchronised"
cond_keys["Intercept"] += ")"

keys_cond = {v:k for k,v in cond_keys.items()}

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()
epo = mne.read_epochs(proc_dir+"grand_central-epo.fif")

# get the min-max clustered t values from the pemutations
perm_file = "{}perm_{}_minmax_{}_{}_{}_{}.pickle".format(proc_dir, baseline,
                                                         osc, badsubjs,
                                                         use_group, sync_fact)
with open(perm_file, "rb") as f:
    minmax_ts = pickle.load(f)


e = epo["OscType=='{}'".format(osc)]
e.resample(tfr.info["sfreq"], n_jobs="cuda")
e.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
if osc == "deltO":
    tfr_avg.crop(tmin=-0.75, tmax=0.75)
    epo.crop(tmin=-0.75, tmax=0.75)
# calculate global ERP min and max for scaling later on
evo = e.average()
ev_min, ev_max = evo.data.min(), evo.data.max()
# get osc ERP and normalise
evo_data = evo.data
evo_data = (evo_data - ev_min) / (ev_max - ev_min)
evo_data = evo_data*5 + 12

stat_conds = list(cond_keys.keys())
tfr_c = tfr_avg.copy()
dat_shape = tfr_c.data.shape[1:]
with open("{}main_fits_{}_grand_{}_{}_{}_{}.pickle".format(proc_dir, baseline,
                                                           osc, badsubjs,
                                                           use_group,
                                                           sync_fact), "rb") as f:
    fits = pickle.load(f)

exog_names = fits["exog_names"]
modfit = fits["fits"]
cks = list(cond_keys.keys())
for order_idx, param_idx in enumerate(range(0,len(cond_keys),9)):
    fig, axes = plt.subplots(3, 3, figsize=(38.4,21.6))
    t_fig, t_axes = plt.subplots(3, 3, figsize=(38.4,21.6))

    ax_w = 4
    mos_array = [["0"]*ax_w + ["1"]*ax_w + ["2"]*ax_w + ["cbar"],
                 ["3"]*ax_w + ["4"]*ax_w + ["5"]*ax_w + ["cbar"],
                 ["6"]*ax_w + ["7"]*ax_w + ["8"]*ax_w + ["cbar"]]
    tfce_fig, tfce_axes = plt.subplot_mosaic(mos_array, figsize=(38.4,21.6))

    axes = [ax for axe in axes for ax in axe]
    t_axes = [ax for axe in t_axes for ax in axe]
    for en_idx,en in enumerate(cks[param_idx:param_idx+9]):
        data = np.zeros((3, len(modfit)))
        for mf_idx, mf in enumerate(modfit):
            data[0, mf_idx] = mf.params[exog_names.index(en)]
            data[1, mf_idx] = mf.tvalues[exog_names.index(en)]
            data[2, mf_idx] = mf.pvalues[exog_names.index(en)]

        pvals = data[2,].reshape(*dat_shape, order="F")
        pvals[np.isnan(pvals)] = 1
        mask = pvals<0.05
        if "Intercept" in en:
            mask = None

        # parameters
        dat = data[0,].reshape(*dat_shape, order="F")
        dat[np.isnan(dat)] = 0
        tfr_c.data[0,] = dat
        tfr_c.plot(picks="central", axes=axes[en_idx], colorbar=False,
                   vmin=vmin, vmax=vmax, cmap="viridis", mask=mask,
                   mask_style="contour")
        axes[en_idx].plot(tfr.times, evo_data[0,],
                          color="gray", alpha=0.8,
                          linewidth=10)
        axes[en_idx].set_title(cond_keys[en])

        # t values
        dat = data[1,].reshape(*dat_shape, order="F")
        tfr_c.data[0,] = dat
        tfr_c.plot(picks="central", axes=t_axes[en_idx], colorbar=False,
                   vmin=vmin, vmax=vmax, cmap="viridis", mask=mask,
                   mask_style="contour")
        t_axes[en_idx].plot(tfr.times, evo_data[0,],
                          color="gray", alpha=0.8,
                          linewidth=10)
        t_axes[en_idx].set_title(cond_keys[en])


        ## parameters corrected for multiple comparisons
        # positive and negative thresholds
        pos_thresh = np.quantile(minmax_ts[en]["max"], 1-perm_thresh)
        neg_thresh = np.quantile(minmax_ts[en]["min"], perm_thresh)

        dat = data[0,].reshape(*dat_shape, order="F")
        t_dat = tfce_correct(data[1,].reshape(*dat_shape, order="F"),tfce_thresh)
        dat[np.isnan(dat)] = 0
        if interp:
            x = np.arange(dat.shape[1])
            y = np.arange(dat.shape[0])
            interper = interp2d(x, y, dat)
            dat = interper(x, y)

        mask_pos = t_dat > pos_thresh
        mask_neg = t_dat < neg_thresh
        mask = mask_pos + mask_neg

        if cond_keys[en] == "Fixed frequency":
            sig_mask = mask # we'll need this for the figure later down

        tfr_c.data[0,] = dat
        tfr_c.plot(picks="central", axes=tfce_axes[str(en_idx)], colorbar=False,
                   vmin=vmin, vmax=vmax, cmap=cmap, mask=mask,
                   mask_style="contour")
        tfce_axes[str(en_idx)].plot(tfr.times, evo_data[0,],
                          color="gray", alpha=0.8,
                          linewidth=10)
        tfce_axes[str(en_idx)].set_title(cond_keys[en])

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, cax=tfce_axes["cbar"])
        tfce_axes["cbar"].set_ylabel("Z-score from baseline", fontsize=28)

        if en_idx % 3 == 0:
            tfce_axes[str(en_idx)].set_ylabel("Frequency (Hz)", fontsize=30)
        else:
            tfce_axes[str(en_idx)].set_ylabel("")
            tfce_axes[str(en_idx)].set_yticks([])
        if en_idx >= 6:
            tfce_axes[str(en_idx)].set_xlabel("Time (s)", fontsize=30)
        else:
            tfce_axes[str(en_idx)].set_xlabel("")
            tfce_axes[str(en_idx)].set_xticks([])

        suptitle_str = "LME estimated parameters of {} spindle power".format(osc)
        plt.suptitle(suptitle_str)
        plt.tight_layout()
        tfce_fig.savefig("../images/lmmtfr_grand_tfce_{}_{}_{}_{}_{}.tif".format(osc, badsubjs, use_group, sync_fact, order_idx))
        tfce_fig.savefig("../images/lmmtfr_grand_tfce_{}_{}_{}_{}_{}.svg".format(osc, badsubjs, use_group, sync_fact, order_idx))

# predictions
pred_cmap = "inferno"
pred_vmin, pred_vmax = 0, 4
coe_keys = list(cond_exogs.keys())
for order_idx, param_idx in enumerate(range(0,len(cond_exogs.keys()),9)):
    ax_w = 4
    mos_array = [["0"]*ax_w + ["1"]*ax_w + ["2"]*ax_w + ["cbar"],
                 ["3"]*ax_w + ["4"]*ax_w + ["5"]*ax_w + ["cbar"],
                 ["6"]*ax_w + ["7"]*ax_w + ["8"]*ax_w + ["cbar"]]
    fig, axes = plt.subplot_mosaic(mos_array, figsize=(38.4,21.6))
    for cond_idx, exog_key in enumerate(coe_keys[param_idx:param_idx+9]):
        data = np.zeros(len(modfit))
        cond_vec = cond2vec(exog_names, cond_exogs[exog_key], keys_cond)
        for mf_idx, mf in enumerate(modfit):
            data[mf_idx] = mf.predict(cond_vec)
        data = data.reshape(*dat_shape, order="F")
        data[np.isnan(data)] = 0
        tfr_c.data[0,] = data
        tfr_c.plot(picks="central", axes=axes[str(cond_idx)], colorbar=False,
                   vmin=vmin, vmax=vmax, cmap=pred_cmap)
        axes[str(cond_idx)].plot(tfr.times, evo_data[0,],
                            color="gray", alpha=0.8,
                            linewidth=10)
        axes[str(cond_idx)].set_title(exog_key)

        norm = matplotlib.colors.Normalize(vmin=pred_vmin, vmax=pred_vmax)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=pred_cmap)
        plt.colorbar(sm, cax=axes["cbar"])
        axes["cbar"].set_ylabel("Z-score from baseline", fontsize=28)

        if cond_idx % 3 == 0:
            axes[str(cond_idx)].set_ylabel("Frequency (Hz)", fontsize=30)
        else:
            axes[str(cond_idx)].set_ylabel("")
            axes[str(cond_idx)].set_yticks([])
        if cond_idx >= 6:
            axes[str(cond_idx)].set_xlabel("Time (s)", fontsize=30)
        else:
            axes[str(cond_idx)].set_xlabel("")
            axes[str(cond_idx)].set_xticks([])

    fig.suptitle("LME model predictions of {} spindle power".format(osc))
    fig.tight_layout()
    fig.savefig("../images/lmmtfr_grand_predict_{}_{}_{}_{}_{}.tif".format(osc, badsubjs, use_group, sync_fact, order_idx))
    fig.savefig("../images/lmmtfr_grand_predict_{}_{}_{}_{}_{}.svg".format(osc, badsubjs, use_group, sync_fact, order_idx))

# 'nice' figure
pred_cmap = "inferno"
pred_vmin, pred_vmax = 0, 4
ax_w = 4
mos_array = [["blank1"]*ax_w + ["fix"]*ax_w + ["fix-sham"]*ax_w + ["cbar"],
             ["sham"]*ax_w + ["fix"]*ax_w + ["fix-sham"]*ax_w + ["cbar"],
             ["sham"]*ax_w + ["eig"]*ax_w + ["eig-sham"]*ax_w + ["cbar"],
             ["blank2"]*ax_w + ["eig"]*ax_w + ["eig-sham"]*ax_w + ["cbar"]]
fig, axes = plt.subplot_mosaic(mos_array, figsize=(38.4,21.6))
coe_keys = {"sham":["Intercept (Sham 30s)"],
            "fix":["Intercept (Sham 30s)", "Fixed frequency"],
            "eig":["Intercept (Sham 30s)", "Eigenfrequency"],
            "fix-sham":["Fixed frequency"],
            "eig-sham":["Eigenfrequency"]}
for k,v in coe_keys.items():
    mask = sig_mask if k == "fix-sham" else np.zeros(mask.shape, dtype="bool")
    data = np.zeros(len(modfit))
    cond_vec = cond2vec(exog_names, v, keys_cond)
    for mf_idx, mf in enumerate(modfit):
        data[mf_idx] = mf.predict(cond_vec)
    data = data.reshape(*dat_shape, order="F")
    data[np.isnan(data)] = 0
    tfr_c.data[0,] = data
    tfr_c.plot(picks="central", axes=axes[k], colorbar=False,
               vmin=vmin, vmax=vmax, cmap=pred_cmap, mask=mask, mask_style="contour")
    axes[k].plot(tfr.times, evo_data[0,],
                 color="gray", alpha=0.8,
                 linewidth=10)
    axes[k].set_title(exog_key)

    norm = matplotlib.colors.Normalize(vmin=pred_vmin, vmax=pred_vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=pred_cmap)
    plt.colorbar(sm, cax=axes["cbar"])
    axes["cbar"].set_ylabel("Z-score from baseline", fontsize=28)

axes["sham"].set_ylabel("Frequency (Hz)", fontsize=30)
axes["sham"].set_xlabel("Time (s)", fontsize=30)
axes["fix-sham"].set_yticks([])
axes["eig-sham"].set_yticks([])
axes["fix"].set_xticks([])
axes["fix"].set_xlabel("")
axes["fix"].set_ylabel("Frequency (Hz)", fontsize=30)
axes["fix-sham"].set_xticks([])
axes["fix-sham"].set_xlabel("")
axes["fix-sham"].set_ylabel("")
axes["eig"].set_xlabel("Time (s)", fontsize=30)
axes["eig"].set_ylabel("Frequency (Hz)", fontsize=30)
axes["eig-sham"].set_xlabel("Time (s)", fontsize=30)
axes["eig-sham"].set_ylabel("")
axes["blank1"].axis("off")
axes["blank2"].axis("off")
axes["sham"].set_title("Sham")
axes["eig"].set_title("Eigen frequency")
axes["fix"].set_title("Fixed frequency")
axes["fix-sham"].set_title("Fixed - Sham")
axes["eig-sham"].set_title("Eigen - Sham")

axes["blank1"].set_title("LME model predictions of {} spindle power".format(osc),
                         fontsize=36)
fig.suptitle("")
fig.tight_layout()
fig.savefig("../images/lmmtfr_fig1_{}_{}_{}_{}_{}.tif".format(osc, badsubjs, use_group, sync_fact, order_idx))
fig.savefig("../images/lmmtfr_fig1_{}_{}_{}_{}_{}.svg".format(osc, badsubjs, use_group, sync_fact, order_idx))
