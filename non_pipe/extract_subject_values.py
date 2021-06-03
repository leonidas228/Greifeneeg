import mne
from mne.time_frequency import read_tfrs
from os.path import isdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)
import seaborn as sns

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

method = "wavelet"
conds = ["sham", "eig", "fix"]
bad_subjs = ["002", "003", "028", "014", "051"]
bad_durs = ["5m"]
baseline = "nobl"
time_win = (250, 650)
freq_win = (12, 15)

sig_mask = np.load("{}sig_mask_Fixed frequency.npy".format(proc_dir))

infile = "{}ModIdx_{}_{}_{}-{}Hz_{}-{}ms.pickle".format(proc_dir, method,
                                                        baseline, *freq_win,
                                                        *time_win)
df_pac = pd.read_pickle(infile)

# infile = "{}kl_divs.pickle".format(proc_dir)
# df_kl = pd.read_pickle(infile)
# kl_subjs = list(df_kl["Subj"].unique())

# load TFR
tfr = read_tfrs("{}grand_central_zscore-tfr.h5".format(proc_dir))[0]
tfr = tfr["PrePost=='Post'"]
# remove all subjects with missing conditions or not meeting synchronicity criterion
for bs in bad_subjs:
    print("Removing subject {}".format(bs))
    tfr = tfr["Subj!='{}'".format(bs)]
    df_pac = df_pac.query("Subj!='{}'".format(bs))
for bd in bad_durs:
    print("Removing duration {}".format(bd))
    tfr = tfr["Dur!='{}'".format(bd)]
    df_pac = df_pac.query("Dur!='{}'".format(bd))

tfr_subjs = list(tfr.metadata["Subj"].unique())
pac_subjs = list(df_pac["Subj"].unique())
subjs = list(set(pac_subjs + tfr_subjs))
subjs.sort()

vars = ["TFR", "PAC"]#, "KLDiv_gauss", "KLDiv_NestSO", "KLDiv_NestDO"]
df_dict = {"Subj":[], "Sync":[]}
for var in vars:
    for cond in conds:
        df_dict["{}_{}".format(var, cond)] = []

for subj in subjs:
    df_dict["Subj"].append(subj)
    # sync
    sync = tfr["Subj=='{}'".format(subj)].metadata.iloc[0]["Sync"]
    df_dict["Sync"].append(sync)
    for cond in conds:
        print("{} {}".format(subj, cond))
        # TFR
        TFR = tfr["Subj=='{}' and StimType=='{}'".format(subj, cond)].average()
        value = TFR.data[0,sig_mask].mean()
        df_dict["TFR_{}".format(cond)].append(value)

        # PAC
        this_df = df_pac.query("Subj=='{}' and StimType=='{}'".format(subj, cond))
        nd_value = this_df["ND"].values.mean()
        df_dict["PAC_{}".format(cond)].append(nd_value)

        # # KL Measures
        # this_df = df_kl.query("Subj=='{}' and Cond=='{}'".format(subj, cond))
        # div_gauss = this_df["gauss_div"].values.mean()
        # div_nestso = this_df["SO_div"].values.mean()
        # div_nestdo = this_df["deltO_div"].values.mean()
        # df_dict["KLDiv_gauss_{}".format(cond)].append(div_gauss)
        # df_dict["KLDiv_NestSO_{}".format(cond)].append(div_nestso)
        # df_dict["KLDiv_NestDO_{}".format(cond)].append(div_nestdo)

df = pd.DataFrame.from_dict(df_dict)

#normalize the columns
# df_norm = df.copy()
# for col in df.keys():
#     if col == "Subj":
#         continue
#     df_norm[col] = (df_norm[col]-df_norm[col].mean())/df_norm[col].std()
# df = df_norm

if bad_durs:
    df.to_csv("../indiv_data/indiv_measures_{}_{}-{}Hz_{}-{}_no{}.csv".format(baseline,
                                                                              *freq_win,
                                                                              *time_win,
                                                                              bad_durs))
else:
    df.to_csv("../indiv_data/indiv_measures_{}_{}-{}Hz_{}-{}.csv".format(baseline,
                                                                 *freq_win,
                                                                 *time_win))


# g = sns.PairGrid(df)
# g.map(sns.scatterplot)
