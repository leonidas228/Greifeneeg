from os.path import isdir
import pandas as pd
import numpy as np
import mne
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 48}
matplotlib.rc('font', **font)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

conds = ["sham", "eig", "fix"]
durs = ["30s", "2m", "5m"]

exclude = ["002", "003", "028", "007", "051"]
#exclude.extend(["045", "038", "027", "046", "033", "053", "017", "022", "009"]) # these are missing conditions, which causes problems with subject averaging
epo = mne.read_epochs("{}grand_central_finfo-epo.fif".format(proc_dir),
                      preload=True)
for excl in exclude:
    epo = epo["Subj!='{}'".format(excl)]
epo.crop(tmin=0, tmax=0.75)
data = epo.get_data()[:,0,]
md = epo.metadata

df_dict = {"Subj":[], "StimType":[], "Sync":[], "Amp":[]}
subjs = list(epo.metadata["Subj"].unique())
for subj in subjs:
    for cond in conds:
        #for dur in durs:
        mask = (md["Subj"]==subj) & (md["StimType"]==cond)# & (md["Dur"]==dur)
        dat = data[mask,].mean(axis=0)
        amp = dat.max() - dat[0]
        sync = md[mask]["Sync"].iloc[0]
        df_dict["Subj"].append(subj)
        df_dict["Sync"].append(sync)
        df_dict["StimType"].append(cond)
        #df_dict["Dur"].append(dur)
        df_dict["Amp"].append(amp)
df = pd.DataFrame.from_dict(df_dict)

groups = df["Subj"]
vc_form = {"Subj": "0 + C(Subj)"}
re_form = "0 + StimType"
formula = "Amp ~ C(StimType, Treatment('sham'))"

mod = smf.mixedlm(formula, df, groups=groups)
mf = mod.fit()
print(mf.summary())
