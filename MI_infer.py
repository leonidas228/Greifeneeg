from os.path import isdir
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 24}
matplotlib.rc('font', **font)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

method="wavelet"
baseline = "nobl"
#baseline = "zscore"
time_win = (150,700)
#time_win = (50,600)

infile = "{}ModIdx_{}_{}_{}-{}ms.pickle".format(proc_dir, method, baseline,
                                                *time_win)
print(infile)
df = pd.read_pickle(infile)

for osc in ["SO"]:
    for var in ["MVL", "ND"]:
        fig, ax = plt.subplots(figsize=(38.4, 21.6))
        this_df = df.query("OscType=='{}'".format(osc))

        sns.barplot(data=this_df, y=var, x="StimType", hue="Dur",
                    order=["sham", "eig", "fix"], hue_order=["30s", "2m", "5m"],
                    ax=ax)
        plt.suptitle("{} {} ({} transformed)".format(osc, var, method))
        plt.savefig("../images/{}_{}_{}".format(var, osc, method))

        vc_form = {"Subj": "0 + C(Subj)"}
        re_form = "0 + Stim"
        re_form = None
        formula = "{} ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))".format(var)
        mod = smf.mixedlm(formula, data=this_df, groups=this_df["Sync"],
                          re_formula=re_form, vc_formula=vc_form)
        mf = mod.fit()
        print(mf.summary())
