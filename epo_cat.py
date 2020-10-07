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
chans = ["frontal", "posterior"]
chans = ["central"]
df_dict = {"Subj":[],"Ort":[],"Cond":[],"OscType":[],"PrePost":[],"Number":[]}

epos = []
epos_ba = []
for filename in filelist:
    this_match = re.match("d_NAP_(\d{3})_(.*)-epo.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        epo = mne.read_epochs(proc_dir+filename)
        if cond == "sham":
            breakpoint()
        max_ind = epo.metadata["Index"].max()
        max_ind = 4 if max_ind > 4 else max_ind
        if max_ind < 2:
            continue
        epo.pick_channels(chans)
        epos.append(epo)
        epo_ba = epo["(PrePost=='Pre' and Index=='0') or (PrePost=='Post' and Index=='{}')".format(max_ind)]
        if len(epo_ba.events):
            epos_ba.append(epo_ba)
        for ort in chans:
            for osc in ["SO","deltO"]:
                for pp in ["Pre","Post"]:
                    df_dict["Subj"].append(subj)
                    df_dict["Cond"].append(cond)
                    df_dict["OscType"].append(osc)
                    df_dict["PrePost"].append(pp)
                    df_dict["Ort"].append(ort)
                    df_dict["Number"].append(len(epo_ba["PrePost=='{}' and OscType=='{}' and Ort=='{}'".format(pp,osc,ort)]))
grand_epo = mne.concatenate_epochs(epos)
grand_epo.save("{}grand-epo.fif".format(proc_dir), overwrite=True)
grand_epo_ba = mne.concatenate_epochs(epos_ba)
grand_epo_ba.save("{}grand_ba-epo.fif".format(proc_dir), overwrite=True)
df = pd.DataFrame.from_dict(df_dict)
df.to_pickle("{}grand_ba_df.pickle".format(proc_dir))
