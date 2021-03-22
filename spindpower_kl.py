import pickle
import numpy as np
from os.path import isdir
from os import listdir
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

def gauss(x, A, mu, sigma):
    y =  A*np.exp(-1.0*(x - mu)**2 / (2*sigma**2))
    return y

def KL_div(p, q):
    p += 1e-5
    q += 1e-5
    kl = np.sum(p * np.log(p/q))
    return kl

def df_prob_dists(df, method="kl"):
    boot_num = df["boot_num"]
    bins = np.array(list(df["Bins"].values))
    bins = np.average(bins, axis=0, weights=boot_num)
    SO_counts = np.array(list(df["SO_counts"].values))
    SO_counts = np.average(SO_counts, axis=0, weights=boot_num)
    deltO_counts = np.array(list(df["deltO_counts"].values))
    deltO_counts = np.average(deltO_counts, axis=0, weights=boot_num)
    free_counts = np.array(list(df["free_counts"].values))
    free_counts = np.average(free_counts, axis=0, weights=boot_num)

    # fitted gaussian
    all_counts = np.vstack([SO_counts, deltO_counts, free_counts]).mean(axis=0)
    (a, mu, std), _ = curve_fit(gauss, bins, all_counts)
    fitted_gauss = gauss(bins, a, mu, std)

    if method == "kl":
        SO_div = KL_div(SO_counts, free_counts)
        deltO_div = KL_div(deltO_counts, free_counts)
        gauss_div = KL_div(all_counts, fitted_gauss)
    elif method == "wass":
        SO_div = wasserstein_distance(free_counts, SO_counts)
        deltO_div = wasserstein_distance(free_counts, deltO_counts)
        gauss_div = wasserstein_distance(fitted_gauss, all_counts)

    ratio = SO_div / deltO_div

    return (SO_div, deltO_div, ratio, gauss_div)

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

perm_n = 1000
conds = ["sham", "fix", "eig"]
exclude = ["002", "003", "028"]

filelist = listdir(proc_dir)

df_dict = {"Subj":[], "Cond":[], "StimType":[], "Dur":[], "Bins":[], "SO_counts":[],
           "deltO_counts":[], "free_counts":[], "boot_num":[]}
for filename in filelist:
    # load files, merge SO and deltO annotations
    match_str = "spindle_distros_(.*)_(.*)_(.*).pickle"
    this_match = re.match(match_str, filename)
    if not this_match:
        continue
    subj, cond, chan = (this_match.group(1), this_match.group(2),
                        this_match.group(3))
    if subj in exclude:
        print("Skipping subject {}".format(subj))
        continue
    with open(proc_dir+filename, "rb") as f:
        histos = pickle.load(f)

        if "fix" in cond:
            stim_type = "fix"
        elif "eig" in cond:
            stim_type = "eig"
        else:
            stim_type = "sham"

        if "30s" in cond:
            dur = "30s"
        elif "2m" in cond:
            dur = "2m"
        else:
            dur = "5m"

        df_dict["Subj"].append(subj)
        df_dict["Cond"].append(cond)
        df_dict["StimType"].append(stim_type)
        df_dict["Dur"].append(dur)
        df_dict["Bins"].append(histos["bin_edges"])
        df_dict["SO_counts"].append(histos["SO_counts"])
        df_dict["deltO_counts"].append(histos["deltO_counts"])
        df_dict["free_counts"].append(histos["free_counts"])
        df_dict["boot_num"].append(histos["boot_num"])

df = pd.DataFrame.from_dict(df_dict)
subjs = list(df["Subj"].unique())
conds = list(df["StimType"].unique())
durs = list(df["Dur"].unique())

kl_dict = {"Subj":[], "Cond":[], "Dur":[], "SO_div":[], "deltO_div":[],
           "gauss_div":[], "Ratio":[], "Sync":[]}
for subj in subjs:
    for cond in conds:
        for dur in durs:
            q_str = "Subj=='{}' and StimType=='{}' and Dur=='{}'".format(subj,
                                                                         cond,
                                                                         dur)
            this_df = df.query(q_str)
            if not len(this_df):
                continue
            SO_kl, deltO_kl, ratio, gauss_kl = df_prob_dists(this_df,
                                                             method="kl")
            if int(subj) < 31 and subj != "021" and subj!='017':
                sync = "async"
            else:
                sync = "sync"

            kl_dict["Subj"].append(subj)
            kl_dict["Cond"].append(cond)
            kl_dict["Dur"].append(dur)
            kl_dict["SO_div"].append(SO_kl)
            kl_dict["deltO_div"].append(deltO_kl)
            kl_dict["gauss_div"].append(gauss_kl)
            kl_dict["Ratio"].append(ratio)
            kl_dict["Sync"].append(sync)
kl_df = pd.DataFrame.from_dict(kl_dict)
kl_df.to_pickle("{}kl_divs.pickle".format(proc_dir))

plt.figure()
sns.barplot(data=kl_df, x="Cond", y="SO_div", hue="Dur",
            order=["sham", "eig", "fix"], hue_order=["30s", "2m", "5m"])
plt.title("SO Nesting")

plt.figure()
sns.barplot(data=kl_df, x="Cond", y="deltO_div", hue="Dur",
            order=["sham", "eig", "fix"], hue_order=["30s", "2m", "5m"])
plt.title("deltO Nesting")

plt.figure()
sns.barplot(data=kl_df, x="Cond", y="Ratio", hue="Dur",
            order=["sham", "eig", "fix"], hue_order=["30s", "2m", "5m"])
plt.title("SO/deltO ratio")

plt.figure()
sns.barplot(data=kl_df, x="Cond", y="gauss_div", hue="Dur",
            order=["sham", "eig", "fix"], hue_order=["30s", "2m", "5m"])
plt.title("Spindle presence")

re_form = None
vc_form = {"Subj": "0 + C(Subj)"}
groups = kl_df["Sync"]
formula = "SO_div ~ C(Cond, Treatment('sham'))*C(Dur, Treatment('30s'))"
md = smf.mixedlm(formula, kl_df, re_formula=re_form,
                 vc_formula=vc_form, groups=groups)
mf = md.fit()
print(mf.summary())

formula = "deltO_div ~ C(Cond, Treatment('sham'))*C(Dur, Treatment('30s'))"
md = smf.mixedlm(formula, kl_df, re_formula=re_form,
                 vc_formula=vc_form, groups=groups)
mf = md.fit()
print(mf.summary())

formula = "Ratio ~ C(Cond, Treatment('sham'))*C(Dur, Treatment('30s'))"
md = smf.mixedlm(formula, kl_df, re_formula=re_form,
                 vc_formula=vc_form, groups=groups)
mf = md.fit()
print(mf.summary())

formula = "gauss_div ~ C(Cond, Treatment('sham'))*C(Dur, Treatment('30s'))"
md = smf.mixedlm(formula, kl_df, re_formula=re_form,
                 vc_formula=vc_form, groups=groups)
mf = md.fit()
print(mf.summary())
