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

time_win = (-0.025, 0.025)
sham = True
chan = "central"

df_SO = pd.read_pickle("{}phase_amp_{}".format(proc_dir, "SO"))
df_deltO = pd.read_pickle("{}phase_amp_{}".format(proc_dir, "deltO"))
df = pd.concat([df_SO, df_deltO])
sub_inds = df["Subj"].values.astype(int) >= 31
df["Synchron"] = sub_inds

df = df[df["Synchron"]]
#df = df.query("Cond=='eig30s' or Cond=='fix30s' or Cond=='sham'")

if sham:
    # # all oscillations
    # this_df = df.copy()
    # #md = smf.mixedlm("Amp ~ C(OscType, Treatment('deltO'))*C(Cond, Treatment('sham'))", this_df, groups=this_df["Subj"])
    # md = smf.mixedlm("Amp ~ C(OscType, Treatment('deltO')) * PureIndex * C(Cond, Treatment('sham'))", this_df, groups=this_df["Subj"])
    # res_all = md.fit(reml=True)
    # print(res_all.summary())

    # SO only
    this_df = df.query("OscType=='SO'")
    md = smf.mixedlm("Amp ~ C(PrePost, Treatment('Pre'))*Index*C(Cond, Treatment('sham'))", this_df,
                     groups=this_df["Subj"])
    res_SO = md.fit(reml=False)
    print(res_SO.summary())
    #
    # # deltO only
    # this_df = df.query("OscType=='deltO'")
    # md = smf.mixedlm("Amp ~ C(PrePost, Treatment('Pre'))*Index*C(Cond, Treatment('sham'))", this_df,
    #                  groups=this_df["Subj"])
    # res_deltO = md.fit(reml=False)
    # print(res_deltO.summary())
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
    md = smf.mixedlm("Amp ~ OscType*PrePost*Index*StimArt*StimLen", this_df,
                     groups=this_df["Subj"])
    res_all = md.fit()
    print(res_all.summary())

    # SO only
    this_df = df.query("OscType=='SO'")
    md = smf.mixedlm("Amp ~ PrePost*Index*StimArt*StimLen", this_df,
                     groups=this_df["Subj"])
    res_SO = md.fit()
    print(res_SO.summary())

    # deltO only
    this_df = df.query("OscType=='deltO'")
    md = smf.mixedlm("Amp ~ PrePost*Index*StimArt*StimLen", this_df,
                     groups=this_df["Subj"])
    res_deltO = md.fit()
    print(res_deltO.summary())
