import mne
import numpy as np
import pandas as pd
from os.path import isdir
import statsmodels.formula.api as smf

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

epo = mne.read_epochs("{}grand_central_finfo_SI-epo.fif".format(proc_dir))
df = epo.metadata

subjs = list(df["Subj"].unique())
subjs.sort()
stim_types = list(df["StimType"].unique())
durs = list(df["Dur"].unique())
osc_types = list(df["OscType"].unique())

SI_df_dict = {"Subj":[], "Sync":[], "OscType":[], "Cond":[], "StimType":[], "Dur":[], "SI_norm":[]}
for subj in subjs:
    for osc in osc_types:
        for stim_type in stim_types:
            for dur in durs:
                this_df = df.query("Subj=='{}' and OscType=='{}' and StimType=='{}' and Dur=='{}'".format(subj, osc, stim_type, dur))
                if len(this_df) > 10:
                    SIs = this_df["SI"].values
                    x_bar, y_bar = np.cos(SIs).mean(), np.sin(SIs).mean()
                    r_norm = np.linalg.norm((x_bar, y_bar))
                    SI_df_dict["Subj"].append(subj)
                    if int(subj) < 31:
                        SI_df_dict["Sync"].append("async")
                    else:
                        SI_df_dict["Sync"].append("sync")
                    SI_df_dict["OscType"].append(osc)
                    SI_df_dict["Cond"].append(stim_type+dur)
                    SI_df_dict["StimType"].append(stim_type)
                    SI_df_dict["Dur"].append(dur)
                    SI_df_dict["SI_norm"].append(r_norm)
SI_df = pd.DataFrame.from_dict(SI_df_dict)
