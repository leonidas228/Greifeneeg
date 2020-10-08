import mne
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

oscs = ["SO", "deltO"]
conds = ["eig30s", "fix30s"]
time_win = (-0.05, 0.05)
sham = True

epo = mne.read_epochs("{}grand-epo.fif".format(proc_dir), preload=True)
df = epo.metadata
sub_inds = df["Subj"].values.astype(int) >= 31

epo.filter(l_freq=0.3,h_freq=3,n_jobs=8)
data = epo.get_data()
time_inds = (epo.time_as_index(time_win[0])[0],epo.time_as_index(time_win[1])[0])
erp = data[:,0,time_inds[0]:time_inds[1]].mean(axis=1) * 1e+6
df["Brain"] = pd.Series(erp,index=df.index)

if sham:
    # all oscillations
    this_df = df.copy()
    md = smf.mixedlm("Brain ~ PrePost*C(Cond, Treatment('sham'))*OscType*Index", this_df,
                     groups=this_df["Subj"])
    res_all = md.fit()
    print(res_all.summary())

    # SO only
    this_df = df.query("OscType=='SO'")
    md = smf.mixedlm("Brain ~ PrePost*Index*C(Cond, Treatment('sham'))", this_df,
                     groups=this_df["Subj"])
    res_SO = md.fit()
    print(res_SO.summary())

    # deltO only
    this_df = df.query("OscType=='deltO'")
    md = smf.mixedlm("Brain ~ PrePost*Index*C(Cond, Treatment('sham'))", this_df,
                     groups=this_df["Subj"])
    res_deltO = md.fit()
    print(res_deltO.summary())
else:
    stim_arts = []
    stim_lens = []
    for cond in list(df["Cond"].values):
        stim_art = "eig" if "eig" in cond else "fix"
        for sl in ["30s","2m","5m"]:
            if sl in cond:
                 stim_len = sl
                 break
            else:
                stim_len = None
        stim_arts.append(stim_art)
        stim_lens.append(stim_len)
    df["StimArt"] = pd.Series(stim_arts, index=df.index)
    df["StimLen"] = pd.Series(stim_lens, index=df.index)

    # all oscillations
    this_df = df.copy()
    md = smf.mixedlm("Brain ~ PrePost + Index + StimArt + StimLen + OscType*Index", this_df,
                     groups=this_df["Subj"])
    res_all = md.fit()
    print(res_all.summary())

    # SO only
    this_df = df.query("OscType=='SO'")
    md = smf.mixedlm("Brain ~ PrePost + Index + StimArt + StimLen", this_df,
                     groups=this_df["Subj"])
    res_SO = md.fit()
    print(res_SO.summary())

    # deltO only
    this_df = df.query("OscType=='deltO'")
    md = smf.mixedlm("Brain ~ PrePost + Index + StimArt + StimLen", this_df,
                     groups=this_df["Subj"])
    res_deltO = md.fit()
    print(res_deltO.summary())
