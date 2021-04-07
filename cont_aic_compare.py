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
import seaborn as sns

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

osc = "SO"
baseline = "zscore"
use_group = "nogroup"
models = ["Null", "FehlerEig", "Fehler75", "Fehler875"]
bs_name = "no2,3,28"

tfr = read_tfrs("{}grand_central_{}-tfr.h5".format(proc_dir, baseline))[0]
tfr_avg = tfr.average()

# epo = mne.read_epochs(proc_dir+"grand_central-epo.fif")
# e = epo["OscType=='{}'".format(osc)]
# e.resample(tfr.info["sfreq"], n_jobs="cuda")
# e.crop(tmin=tfr.times[0], tmax=tfr.times[-1])
# # calculate global ERP min and max for scaling later on
# evo = e.average()
# ev_min, ev_max = evo.data.min(), evo.data.max()
# # get osc ERP and normalise
# evo_data = evo.data
# evo_data = (evo_data - ev_min) / (ev_max - ev_min)
# evo_data = evo_data*4 + 12

tfr_c = {mod:tfr_avg.copy() for mod in models}
dat_shape = tfr_avg.data.shape[1:]

for mod in models:
    outfile = "{}main_fits_{}_{}_{}_{}_cont_{}.pickle".format(proc_dir,
                                                              baseline, osc,
                                                              bs_name,
                                                              use_group,
                                                              mod)
    with open(outfile, "rb") as f:
        fits = pickle.load(f)
    modfits = fits["fits"]
    aics = np.zeros(len(modfits))
    for mf_idx, mf in enumerate(modfits):
        aics[mf_idx] = mf.aic
    tfr_c[mod].data[0,] = aics.reshape(*dat_shape, order="F")

aics = np.zeros((len(models),*dat_shape))
for mod_idx,mod in enumerate(models):
    aics[mod_idx,] = tfr_c[mod].data
aics[aics==-np.inf] = np.inf
null_compare = aics - aics[0,]

winner_inds = np.argmin(aics, axis=0)
not_null_inds = ((winner_inds != 0) & (np.min(null_compare, axis=0)<-3))



# fig, axes = plt.subplots(1, len(exog_names), figsize=(38.4,21.6))
# #axes = [ax for axe in axes for ax in axe]
# for en_idx,en in enumerate(exog_names):
#     data = np.zeros((3, len(modfit)))
#     for mf_idx, mf in enumerate(modfit):
#         data[0, mf_idx] = mf.params[exog_names.index(en)]
#         data[1, mf_idx] = mf.tvalues[exog_names.index(en)]
#         data[2, mf_idx] = mf.pvalues[exog_names.index(en)]
#         if mf_idx == dat_idx:
#             moi = mf
#
#     pvals = data[2,].reshape(*dat_shape, order="F")
#     pvals[np.isnan(pvals)] = 1
#     mask = pvals<0.05
#     if "Intercept" in en:
#         mask = None
#     dat = data[0,].reshape(*dat_shape, order="F")
#     dat[np.isnan(dat)] = 0
#     tfr_c.data[0,] = dat
#     tfr_c.plot(picks="central", axes=axes[en_idx], colorbar=False, vmin=vmin, vmax=vmax, cmap="viridis", mask=mask, mask_style="contour")
#     axes[en_idx].plot(tfr.times, evo_data[0,],
#                       color="gray", alpha=0.8,
#                       linewidth=10)
#     axes[en_idx].set_title(en)
# plt.tight_layout()
# plt.suptitle("Estimated parameters")
# plt.savefig("../images/lmmtfr_{}_{}_{}_cont_{}.tif".format(osc, bs_name, use_group, cont_var))
