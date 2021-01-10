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
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham30s","sham2m","sham5m"]
filelist = listdir(proc_dir)
chans = ["central"]
osc_types = ["SO", "deltO"]
epo_pref = "ak_"
epo_pref = ""

df_dict = {"Subj":[],"Ort":[],"Cond":[],"OscType":[],"PrePost":[],"Number":[],
           "Index":[], "Stim":[]}
for chan in chans:
    epos = []
    for ot in osc_types:
        for filename in filelist:
            this_match = re.match("d_"+epo_pref+"NAP_(\d{3})_(.*)_(.*)_(.*)-epo.fif", filename)
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
                        if "sham" not in cond:
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
    df = grand_epo.metadata

    # special cases
    drop_inds = np.array([])
    temp = (df["Subj"]=="035").values & (df["Cond"]=="eig5m").values & (df["Index"]>=3).values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))
    temp = (df["Subj"]=="033").values & (df["Cond"]=="eig30s").values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))
    temp = (df["Subj"]=="044").values & (df["Cond"]=="fix2m").values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))
    temp = (df["Subj"]=="046").values & (df["Cond"]=="eig2m").values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))
    temp = (df["Subj"]=="022").values & (df["Cond"]=="eig5m").values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))
    temp = (df["Subj"]=="031").values & (df["Cond"]=="eig5m").values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))
    temp = (df["Subj"]=="046").values & (df["Cond"]=="eig5m").values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))
    temp = (df["Subj"]=="044").values & (df["Cond"]=="eig5m").values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))
    temp = (df["Subj"]=="053").values & (df["Cond"]=="eig5m").values
    drop_inds = np.hstack((drop_inds, np.where(temp)[0]))

    grand_epo.drop(np.array(drop_inds))

    grand_epo = grand_epo["Index < 5"]
    grand_epo.save("{}{}grand_{}-epo.fif".format(proc_dir, epo_pref, chan), overwrite=True)
df = pd.DataFrame.from_dict(df_dict)
df.to_pickle("{}{}grand_df.pickle".format(proc_dir, epo_pref))
