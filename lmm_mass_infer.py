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

perm_chunk = 32
perm_chunk_n = 0

cond_keys = {"Intercept":"Intercept",
             "C(Cond, Treatment('sham'))[T.eig30s]":"eig30s",
             "C(Cond, Treatment('sham'))[T.fix30s]":"fix30s"}

with open("{}main_result.pickle".format(proc_dir), "rb") as f:
    main_result = pickle.load(f)

perms = []
for pcn in range(perm_chunk_n):
    try:
        with open("{}perm_result_{}_{}.pickle".format(proc_dir, perm_chunk, pcn), "rb") as f:
            perms.append(pickle.load(f))
    except:
        continue

stat_conds = list(main_result["raw_t"].keys())
tfce_pos = {cond_keys[k]:[] for k in stat_conds}
tfce_neg = {cond_keys[k]:[] for k in stat_conds}
for perm in perms:
    for pn in perm:
        for sc in stat_conds:
            tfce_pos[cond_keys[sc]].append(pn["tfce_pos"][sc])
            tfce_neg[cond_keys[sc]].append(pn["tfce_neg"][sc])
tfce_pos = {k:np.array(v).ravel() for k,v in tfce_pos.items()}
tfce_neg = {k:np.array(v).ravel() for k,v in tfce_pos.items()}
