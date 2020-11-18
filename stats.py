import mne
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from os.path import isdir
import seaborn as sns

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

# SO only
df_SO_ph = pd.read_pickle("{}phase_amp_{}".format(proc_dir, "SO"))
df_SO_phvar = pd.read_pickle("{}phase_variance_{}".format(proc_dir, "SO"))

df_SO_ph = df_SO_ph.query("PureIndex!='10' and PureIndex!='11'")

sub_inds = df_SO_ph["Subj"].values.astype(int) >= 31
df_SO_ph["Synchron"] = sub_inds

#df_SO_ph = df_SO_ph[df_SO_ph["Synchron"]]
#df_SO_ph = df_SO_ph.query("Cond=='eig30s' or Cond=='fix30s' or Cond=='sham'")

md = smf.mixedlm("SecondAmp ~ C(Cond, Treatment('sham'))", df_SO_ph,
                 groups=df_SO_ph["Subj"])
res_SO = md.fit(reml=False)
print(res_SO.summary())
