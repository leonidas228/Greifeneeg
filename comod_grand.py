import mne
from os import listdir
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
conds = ["eig30s","fix30s"]
filelist = listdir(proc_dir)
chans = ["central"]
chan = "central"
low_fq_range = list(np.linspace(0.5,1.25,50))
high_fq_range =  list(np.linspace(8,20,50))
fs = 100

pacs = []
pps = []
for filename in filelist:
    this_match = re.match("NAP_(\d{3})_(.*)-epo.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds or int(subj) < 31:
            continue
        with open("{}{}_{}_{}_pac.pickle".format(proc_dir, subj, cond, chan), "rb") as f:
            pacs.append(pickle.load(f))
        with open("{}{}_{}_{}_pp.pickle".format(proc_dir, subj, cond, chan), "rb") as f:
            pps.append(pickle.load(f))

p = pacs[0][0]
pacs_arr = np.stack([p[1].mean(axis=-1) for p in pacs])
pacs_grand = np.mean(pacs_arr,axis=0)
p.comodulogram(pacs_grand)

plt.figure()
pp = pps[0][0]
vecs_arr = pps[0][1][2]
amps_arr = np.stack([pp[1][0].mean(axis=-1).squeeze() for pp in pps])
amps_grand = np.mean(amps_arr,axis=0)
vecs_grand = np.mean(vecs_arr,axis=0)
pp.polar(amps_grand.T, vecs_arr, p.yvec)
