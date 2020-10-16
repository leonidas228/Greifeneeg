import mne
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from mne.time_frequency import read_tfrs
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

oscs = ["SO", "deltO"]
conds = ["eig30s", "fix30s"]
time_win = (0.15, 0.5)
sham = True
freq_win = (13, 17)

tfr = read_tfrs("{}grand-tfr.h5".format(proc_dir))[0]
freq_inds = (list(tfr.freqs).index(freq_win[0]), list(tfr.freqs).index(freq_win[1]))
time_inds = (list(tfr.times).index(time_win[0]), list(tfr.times).index(time_win[1]))
df = tfr.metadata
sub_inds = df["Subj"].values.astype(int) >= 31

data = tfr.data
erp = data[:,0,:,time_inds[0]:time_inds[1]].mean(axis=2)
erp = erp[:,freq_inds[0]:freq_inds[1]].mean(axis=1)
df["Brain"] = pd.Series(erp,index=df.index)
df = df[sub_inds]
#df = df.query("Cond=='eig30s' or Cond=='fix30s' or Cond=='sham'")
#df = df.query("Cond=='fix30s' or Cond=='sham'")

if sham:
    # all oscillations
    this_df = df.copy()
    #md = smf.mixedlm("Brain ~ C(OscType, Treatment('deltO'))", this_df, groups=this_df["Subj"])
    #md = smf.mixedlm("Brain ~ C(PrePost, Treatment('Pre'))*C(OscType, Treatment('deltO'))*Index*C(Cond, Treatment('sham'))", this_df, groups=this_df["Subj"])
    md = smf.mixedlm("Brain ~ Index*C(OscType, Treatment('deltO'))*C(Cond, Treatment('sham'))", this_df, groups=this_df["Subj"])
    res_all = md.fit(reml=False)
    print(res_all.summary())

    # SO only
    this_df = df.query("OscType=='SO'")
    md = smf.mixedlm("Brain ~ Index*C(Cond, Treatment('sham'))", this_df, groups=this_df["Subj"])
    res_SO = md.fit(reml=True)
    print(res_SO.summary())

    # deltO only
    this_df = df.query("OscType=='deltO'")
    md = smf.mixedlm("Brain ~ Index*C(Cond, Treatment('sham'))", this_df, groups=this_df["Subj"])
    res_deltO = md.fit(reml=True)
    print(res_deltO.summary())
else:
    df = df.query("Cond != 'sham'")
    stim_arts = []
    stim_lens = []
    for cond in list(df["Cond"].values):
        stim_art = "eig" if "eig" in cond else "fix"
        for sl in ["30s", "2m", "5m"]:
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
    md = smf.mixedlm("Brain ~ OscType*PrePost*Index*StimArt*StimLen", this_df,
                     groups=this_df["Subj"])
    res_all = md.fit()
    print(res_all.summary())

    # SO only
    this_df = df.query("OscType=='SO'")
    md = smf.mixedlm("Brain ~ PrePost*Index*StimArt*StimLen", this_df,
                     groups=this_df["Subj"])
    res_SO = md.fit()
    print(res_SO.summary())

    # deltO only
    this_df = df.query("OscType=='deltO'")
    md = smf.mixedlm("Brain ~ PrePost*Index*StimArt*StimLen", this_df,
                     groups=this_df["Subj"])
    res_deltO = md.fit()
    print(res_deltO.summary())
