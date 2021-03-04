import pickle
from mne.stats.cluster_level import _find_clusters
import numpy as np
from os.path import isdir

def tfce_correct(data, tfce_thresh):
    pos_data = data.copy()
    pos_data[pos_data<0] = 0
    neg_data = data.copy()
    neg_data[neg_data>0] = 0
    pos_clusts = _find_clusters(pos_data, tfce_thresh)[1].reshape(data.shape)
    neg_clusts = _find_clusters(neg_data, tfce_thresh)[1].reshape(data.shape)
    out_data = np.zeros_like(data) + pos_clusts - neg_clusts
    return out_data

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

iter_num = 255
baseline = "zscore"
osc = "SO"
sync_fact = "syncfact"
bad_subjs = "no2,3"
use_group = "group"
tfce_thresh = dict(start=0, step=0.2)
minmax_ts = {}
for iter_idx in range(iter_num):
    infile = "{}perm_{}_grand_{}_{}_{}_{}_{}.pickle".format(proc_dir, baseline,
                                                            osc, bad_subjs,
                                                            use_group, sync_fact,
                                                            iter_idx)
    with open(infile, "rb") as f:
        in_dict = pickle.load(f)
    exog_names = in_dict["exog_names"]
    t_vals = in_dict["t_vals"]
    for en_idx, en in enumerate(exog_names):
        if en not in minmax_ts:
            minmax_ts[en] = {}
            minmax_ts[en]["min"] = []
            minmax_ts[en]["max"] = []
        for perm_idx in range(len(t_vals)):
            this_tfce = tfce_correct(t_vals[perm_idx,en_idx,], tfce_thresh)
            minmax_ts[en]["max"].append(this_tfce.max())
            minmax_ts[en]["min"].append(this_tfce.min())

for en in exog_names:
    minmax_ts[en]["max"] = np.array(minmax_ts[en]["max"])
    minmax_ts[en]["min"] = np.array(minmax_ts[en]["min"])

outfile = "{}perm_{}_minmax_{}_{}_{}_{}.pickle".format(proc_dir, baseline,
                                                          osc, bad_subjs,
                                                          use_group, sync_fact)
with open(outfile, "wb") as f:
    pickle.dump(minmax_ts, f)
