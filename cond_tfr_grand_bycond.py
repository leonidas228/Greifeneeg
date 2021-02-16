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

cond_keys = {"Intercept":{"Name":"Intercept","Loc":(0,0)},
             "C(Dur, Treatment('30s'))[T.2m]":{"Name":"2m","Loc":(0,1)},
             "C(Dur, Treatment('30s'))[T.5m]":{"Name":"5m","Loc":(0,2)},
             "C(PrePost, Treatment('Pre'))[T.Post]":{"Name":"Post", "Loc":(1,0)},
             "C(Dur, Treatment('30s'))[T.2m]:C(PrePost, Treatment('Pre'))[T.Post]":{"Name":"2m Post", "Loc":(1,1)},
             "C(Dur, Treatment('30s'))[T.5m]:C(PrePost, Treatment('Pre'))[T.Post]":{"Name":"5m Post", "Loc":(1,2)}
            }

cond_exogs =   {"Sham 30s synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)"],
                "Eigenfrequency 30s synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Eigenfrequency"],
                "Fixed frequency 30s synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Fixed frequency"],
                "Sham 2m synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m"],
                "Eigenfrequency 2m synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Eigenfrequency", "Eigenfrequency 2m"],
                "Fixed frequency 2m synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Fixed frequency", "Fixed frequency 2m"],
                "Sham 5m synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m"],
                "Eigenfrequency 5m synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Eigenfrequency", "Eigenfrequency 5m"],
                "Fixed frequency 5m synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Fixed frequency", "Fixed frequency 5m"],

                "Sham 30s non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "non-synchronised"],
                "Eigenfrequency 30s non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Eigenfrequency", "non-synchronised", "Eigenfrequency non-synchronised"],
                "Fixed frequency 30s non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Fixed frequency", "non-synchronised", "Fixed frequency non-synchronised"],
                "Sham 2m non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "non-synchronised", "2m non-synchronised"],
                "Eigenfrequency 2m non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Eigenfrequency", "non-synchronised", "2m non-synchronised", "Eigenfrequency 2m", "Eigenfrequency non-synchronised", "Eigenfrequency 2m non-synchronised"],
                "Fixed frequency 2m non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Fixed frequency", "non-synchronised", "2m non-synchronised", "Fixed frequency 2m", "Fixed frequency non-synchronised", "Fixed frequency 2m non-synchronised"],
                "Sham 5m non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "non-synchronised", "5m non-synchronised"],
                "Eigenfrequency 5m non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Eigenfrequency", "non-synchronised", "5m non-synchronised", "Eigenfrequency 5m", "Eigenfrequency non-synchronised", "Eigenfrequency 5m non-synchronised"],
                "Fixed frequency 5m non-synchronised pre-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Fixed frequency", "non-synchronised", "5m non-synchronised", "Fixed frequency 5m", "Fixed frequency non-synchronised", "Fixed frequency 5m non-synchronised"],

                "Sham 30s synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Post-stimulation"],
                "Eigenfrequency 30s synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Eigenfrequency", "Post-stimulation", "Eigenfrequency Post-stimulation"],
                "Fixed frequency 30s synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Fixed frequency", "Post-stimulation", "Fixed frequency Post-stimulation"],
                "Sham 2m synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Post-stimulation", "2m Post-stimulation"],
                "Eigenfrequency 2m synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Eigenfrequency", "Post-stimulation", "Eigenfrequency 2m",  "Eigenfrequency Post-stimulation", "2m Post-stimulation", "Eigenfrequency 2m Post-stimulation"],
                "Fixed frequency 2m synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Fixed frequency", "Post-stimulation", "Fixed frequency 2m",  "Fixed frequency Post-stimulation", "2m Post-stimulation", "Fixed frequency 2m Post-stimulation"],
                "Sham 5m synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Post-stimulation", "5m Post-stimulation"],
                "Eigenfrequency 5m synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Eigenfrequency", "Post-stimulation", "Eigenfrequency 5m",  "Eigenfrequency Post-stimulation", "5m Post-stimulation", "Eigenfrequency 5m Post-stimulation"],
                "Fixed frequency 5m synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Fixed frequency", "Post-stimulation", "Fixed frequency 5m",  "Fixed frequency Post-stimulation", "5m Post-stimulation", "Fixed frequency 5m Post-stimulation"],

                "Sham 30s non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "non-synchronised", "Post-stimulation", "Post-stimulation non-synchronised"],
                "Eigenfrequency 30s non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Eigenfrequency", "non-synchronised", "Post-stimulation", "Eigenfrequency non-synchronised", "Eigenfrequency Post-stimulation", "Post-stimulation non-synchronised", "Eigenfrequency Post-stimulation non-synchronised"],
                "Fixed frequency 30s non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "Fixed frequency", "non-synchronised", "Post-stimulation", "Fixed frequency non-synchronised",  "Fixed frequency Post-stimulation", "Post-stimulation non-synchronised", "Fixed frequency Post-stimulation non-synchronised"],
                "Sham 2m non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Post-stimulation", "non-synchronised", "2m Post-stimulation", "Post-stimulation non-synchronised", "2m non-synchronised", "2m Post-stimulation non-synchronised"],
                "Eigenfrequency 2m non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Eigenfrequency", "non-synchronised", "Post-stimulation", "Eigenfrequency 2m", "Eigenfrequency Post-stimulation", "Eigenfrequency non-synchronised", "2m Post-stimulation", "2m non-synchronised", "Post-stimulation non-synchronised", "Eigenfrequency 2m Post-stimulation", "Eigenfrequency 2m non-synchronised", "Eigenfrequency Post-stimulation non-synchronised","2m Post-stimulation non-synchronised", "Eigenfrequency 2m Post-stimulation non-synchronised"],
                "Fixed frequency 2m non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "2m", "Fixed frequency", "non-synchronised", "Post-stimulation", "Fixed frequency 2m", "Fixed frequency Post-stimulation", "Fixed frequency non-synchronised", "2m Post-stimulation", "2m non-synchronised", "Post-stimulation non-synchronised", "Fixed frequency 2m Post-stimulation", "Fixed frequency 2m non-synchronised", "Fixed frequency Post-stimulation non-synchronised","2m Post-stimulation non-synchronised", "Fixed frequency 2m Post-stimulation non-synchronised"],
                "Sham 5m non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Post-stimulation", "non-synchronised", "5m Post-stimulation", "5m non-synchronised", "Post-stimulation non-synchronised", "5m Post-stimulation non-synchronised"],
                "Eigenfrequency 5m non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Eigenfrequency", "non-synchronised", "Post-stimulation", "Eigenfrequency 5m", "Eigenfrequency Post-stimulation", "Eigenfrequency non-synchronised", "5m Post-stimulation", "5m non-synchronised", "Post-stimulation non-synchronised", "Eigenfrequency 5m Post-stimulation", "Eigenfrequency 5m non-synchronised", "Eigenfrequency Post-stimulation non-synchronised","5m Post-stimulation non-synchronised", "Eigenfrequency 5m Post-stimulation non-synchronised"],
                "Fixed frequency 5m non-synchronised post-stimulation":["Intercept (sham30s synchronised pre-stimulation)", "5m", "Fixed frequency", "non-synchronised", "Post-stimulation", "Fixed frequency 5m", "Fixed frequency Post-stimulation", "Fixed frequency non-synchronised", "5m Post-stimulation", "5m non-synchronised", "Post-stimulation non-synchronised", "Fixed frequency 5m Post-stimulation", "Fixed frequency 5m non-synchronised", "Fixed frequency Post-stimulation non-synchronised","5m Post-stimulation non-synchronised", "Fixed frequency 5m Post-stimulation non-synchronised"]
                }


conds = ["sham","fix","eig"]
osc = "SO"
baseline = "logmean"
sync_fact = "syncfact"
use_group = "nogroup"
prepost = True
balance_conds = False
badsubjs = "all_subj"
if baseline == "zscore":
    vmin, vmax = -2.5, 2.5
elif baseline == "logmean":
    vmin, vmax = -.35, .35

if sync_fact == "syncfact":
    # adjust the keys
    new_cond_keys = {}
    for k,v in cond_keys.items():
        if "Intercept" not in k:
            name, loc = v["Name"], v["Loc"]
            new_cond_keys[k+":C(Sync, Treatment('sync'))[T.async]"] = \
                          {"Name":name+" non-synchronised", "Loc":(loc[0]+2,loc[1])}
    cond_keys["C(Sync, Treatment('sync'))[T.async]"] = {"Name":"non-synchronised",
                                                        "Loc":(2,0)}
    cond_keys = {**cond_keys, **new_cond_keys}

cond_keys["Intercept"]["Name"] = "Intercept (Pre-stimulation 30s"
if sync_fact == "syncfact":
    cond_keys["Intercept"]["Name"] += " synchronised"
cond_keys["Intercept"]["Name"] += ")"

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()
epo = mne.read_epochs(proc_dir+"grand_central-epo.fif")
tfr_c = tfr_avg.copy()
dat_shape = tfr_c.data.shape[1:]

e = epo["OscType=='{}'".format(osc)]
e.resample(tfr.info["sfreq"], n_jobs="cuda")
e.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
# calculate global ERP min and max for scaling later on
evo = e.average()
ev_min, ev_max = evo.data.min(), evo.data.max()
# get osc ERP and normalise
evo_data = evo.data
evo_data = (evo_data - ev_min) / (ev_max - ev_min)
evo_data = evo_data*4 + 12

fig_nums = 2 if sync_fact == "nosyncfact" else 4
all_axes = []
for fig_idx in range(fig_nums):
    fig, axes = plt.subplots(3, 3, figsize=(38.4,21.6))
    all_axes.append(axes)


for cond_idx, cond in enumerate(conds):
    mod_file = "{}main_fits_{}_grand_{}_{}_{}_{}_{}.pickle".format(proc_dir,
                                                                   baseline,
                                                                   cond, osc,
                                                                   badsubjs,
                                                                   use_group,
                                                                   sync_fact)

    with open(mod_file, "rb") as f:
        fits = pickle.load(f)
    exog_names = fits["exog_names"]
    modfit = fits["fits"]

    for ck_k, ck_v in cond_keys.items():
        loc, param_name = ck_v["Loc"], ck_v["Name"]
        data = np.zeros((3, len(modfit)))
        for mf_idx, mf in enumerate(modfit):
            data[0, mf_idx] = mf.params[exog_names.index(ck_k)]
            data[1, mf_idx] = mf.tvalues[exog_names.index(ck_k)]
            data[2, mf_idx] = mf.pvalues[exog_names.index(ck_k)]
        pvals = data[2,].reshape(*dat_shape, order="F")
        pvals[np.isnan(pvals)] = 1
        mask = pvals<0.05
        if "Intercept" in param_name:
            mask = None
        dat = data[0,].reshape(*dat_shape, order="F")
        dat[np.isnan(dat)] = 0
        tfr_c.data[0,] = dat
        this_ax = all_axes[loc[0]][loc[1], cond_idx]
        tfr_c.plot(picks="central", axes=this_ax, colorbar=False,
                   vmin=vmin, vmax=vmax, cmap="viridis", mask=mask,
                   mask_style="contour")
        this_ax.plot(tfr.times, evo_data[0,],
                     color="gray", alpha=0.8,
                     linewidth=10)
        this_ax.set_title("{} {}".format(cond, param_name))

        # fig.suptitle("{}_{}_{}_{}".format(osc, bs_name, use_group, sync_fact))
        # if sync_fact == "syncfact":
        #     fig.suptitle("LME model parameters of SO spindle power, synchronicity tested")
        # else:
        #     fig.suptitle("LME model parameters of SO spindle power, synchronicity not tested")
        # fig.tight_layout()
        # fig.savefig("../images/lmmtfr_grand_{}_{}_{}_{}_{}.tif".format(osc, bs_name, use_group, sync_fact, order_idx))


# # predictions
# coe_keys = list(cond_exogs.keys())
# for param_idx in range(0,len(cond_exogs.keys()),9):
#     fig, axes = plt.subplots(3, 3, figsize=(38.4,21.6))
#     axes = [ax for axe in axes for ax in axe]
#     for cond_idx, exog_key in enumerate(coe_keys[param_idx:param_idx+9]):
#         data = np.zeros(len(modfit))
#         cond_vec = cond2vec(exog_names, cond_exogs[exog_key], keys_cond)
#         for mf_idx, mf in enumerate(modfit):
#             data[mf_idx] = mf.predict(cond_vec)
#         data = data.reshape(*dat_shape, order="F")
#         data[np.isnan(data)] = 0
#         tfr_c.data[0,] = data
#         tfr_c.plot(picks="central", axes=axes[cond_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis")
#         axes[cond_idx].plot(tfr.times, evo_data[0,],
#                             color="gray", alpha=0.8,
#                             linewidth=10)
#         axes[cond_idx].set_title(exog_key)
#
# if sync_fact == "syncfact":
#     fig0.suptitle("LME model predictions of SO spindle power, synchronicity tested")
#     fig1.suptitle("LME model predictions of SO spindle power, synchronicity tested")
#     fig1.tight_layout()
#     fig1.savefig("../images/lmmtfr_grand_predict_{}_{}_{}_{}_1.tif".format(osc, bs_name, use_group, sync_fact))
# else:
#     fig0.suptitle("LME model predictions of SO spindle power, synchronicity not tested")
# fig0.tight_layout()
# fig0.savefig("../images/lmmtfr_grand_predict_{}_{}_{}_{}_0.tif".format(osc, bs_name, use_group, sync_fact))
