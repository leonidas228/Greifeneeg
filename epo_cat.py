import mne
from os import listdir
import re
import numpy as np
from os.path import isdir
import pandas as pd

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
#conds = ["eig30s","fix30s"]
filelist = listdir(proc_dir)
chans = ["frontal", "central"]
osc_types = ["SO", "deltO"]

df_dict = {"Subj":[],"Ort":[],"Cond":[],"OscType":[],"PrePost":[],"Number":[],
           "Index":[], "Stim":[]}
for chan in chans:
    epos = []
    for ot in osc_types:
        for filename in filelist:
            this_match = re.match("d_NAP_(\d{3})_(.*)_(.*)_(.*)-epo.fif", filename)
            if this_match:
                print(filename)
                subj, cond = this_match.group(1), this_match.group(2)
                ort, osc_type = this_match.group(3), this_match.group(4)
                if cond not in conds:
                    continue
                if chan != ort or ot != osc_type:
                    continue
                epo = mne.read_epochs(proc_dir+filename)
                max_ind = epo.metadata["Index"].max()
                max_ind = 4 if max_ind > 4 else max_ind
                if max_ind < 2:
                    continue
                epo.pick_channels([chan])
                epos.append(epo)
                for pp in ["Pre","Post"]:
                    for ind in range(max_ind+1):
                        if cond != "sham":
                            df_dict["Stim"].append("Stim")
                        else:
                            df_dict["Stim"].append("Sham")
                        df_dict["Subj"].append(subj)
                        df_dict["Cond"].append(cond)
                        df_dict["OscType"].append(osc_type)
                        df_dict["PrePost"].append(pp)
                        df_dict["Ort"].append(chan)
                        df_dict["Index"].append(ind)
                        df_dict["Number"].append(len(epo["PrePost=='{}' and Index=='{}'".format(pp,ind)]))
    grand_epo = mne.concatenate_epochs(epos)
    grand_epo.save("{}grand_{}-epo.fif".format(proc_dir, chan), overwrite=True)
df = pd.DataFrame.from_dict(df_dict)
df.to_pickle("{}grand_df.pickle".format(proc_dir))
