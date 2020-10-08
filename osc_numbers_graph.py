import mne
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pandas as pd
from os.path import isdir
import seaborn as sns
import statsmodels.formula.api as smf

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

df = pd.read_pickle("{}grand_df.pickle".format(proc_dir))
# sub_inds = df["Subj"].values.astype(int) >= 31
# df = df[sub_inds]

this_df = df.query("OscType=='SO'")
sns.catplot(hue="PrePost", y="Number", x="Cond", data=this_df, kind="violin",
            split=True)
plt.title("Number of slow oscillations")


this_df = df.query("OscType=='deltO'")
sns.catplot(hue="PrePost", y="Number", x="Cond", data=this_df, kind="violin",
            split=True)
plt.title("Number of delta oscillations")

md = smf.mixedlm("Number ~ PrePost*C(Cond, Treatment('sham'))*OscType*Index", df,
                 groups=df["Subj"])
res_all = md.fit()
print(res_all.summary())
