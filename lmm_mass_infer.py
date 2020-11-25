import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pickle
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/perms/"

perm_chunk = 16
perm_chunk_n = 64
model="cond"

cond_keys = {"Intercept":"Intercept",
             "C(Cond, Treatment('sham'))[T.eig30s]":"eig30s",
             "C(Cond, Treatment('sham'))[T.fix30s]":"fix30s",
             "C(Stim, Treatment('sham'))[T.stim]":"stim"}

with open("{}main_result_{}.pickle".format(proc_dir, model), "rb") as f:
    main_result = pickle.load(f)
stat_conds = list(main_result["raw_t"].keys())
main_tfce_pos = {cond_keys[k]:main_result["tfce_pos"][k] for k in stat_conds}
main_tfce_neg = {cond_keys[k]:main_result["tfce_neg"][k] for k in stat_conds}

perms = []
for pcn in range(perm_chunk_n):
    try:
        with open("{}perm_result_{}_{}_{}.pickle".format(proc_dir, perm_chunk, pcn, model), "rb") as f:
            perms.append(pickle.load(f))
    except:
        continue

perm_tfce_pos = {cond_keys[k]:[] for k in stat_conds}
perm_tfce_neg = {cond_keys[k]:[] for k in stat_conds}
for perm in perms:
    for pn in perm:
        for sc in stat_conds:
            perm_tfce_pos[cond_keys[sc]].append(pn["tfce_pos"][sc])
            perm_tfce_neg[cond_keys[sc]].append(pn["tfce_neg"][sc])
perm_tfce_pos = {k:np.array(v).ravel() for k,v in perm_tfce_pos.items()}
perm_tfce_neg = {k:np.array(v).ravel() for k,v in perm_tfce_neg.items()}

pos_thresh = {k:np.percentile(perm_tfce_pos[k],95) for k in perm_tfce_pos.keys()}
neg_thresh = {k:np.percentile(perm_tfce_neg[k],95) for k in perm_tfce_neg.keys()}
