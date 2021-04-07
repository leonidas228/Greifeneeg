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
bad_subjs = ["002", "003", "028"]

sig_mask = np.load("{}sig_mask_Fixed frequency.npy".format(proc_dir))

infile = "{}ModIdx_{}.pickle".format(proc_dir, method)
df_pac = pd.read_pickle(infile)
pac_subjs = list(df_pac["Subj"].unique())

infile = "{}kl_divs.pickle".format(proc_dir)
df_kl = pd.read_pickle(infile)
kl_subjs = list(df_kl["Subj"].unique())

# load TFR
tfr = read_tfrs("{}grand_central_zscore-tfr.h5".format(proc_dir))[0]
tfr = tfr["PrePost=='Post'"]
# remove all subjects with missing conditions or not meeting synchronicity criterion
for bs in bad_subjs:
    print("Removing subject {}".format(bs))
    tfr = tfr["Subj!='{}'".format(bs)]
tfr_subjs = list(tfr.metadata["Subj"].unique())

subjs = list(set(pac_subjs + kl_subjs + tfr_subjs))

df_dict = {"Subj":[], "TFR":[], "PAC":[], "KLDiv_gauss":[], "KLDiv_NestSO":[],
           "KLDiv_NestDO":[]}
for subj in subjs:
    df_dict["Subj"].append(subj)
    # TFR
    sham_TFR = tfr["Subj=='{}' and StimType=='sham'".format(subj)].average()
    sham_value = sham_TFR.data[0,sig_mask].mean()
    fix_TFR = tfr["Subj=='{}' and StimType=='fix'".format(subj)].average()
    fix_value = fix_TFR.data[0,sig_mask].mean()
    df_dict["TFR"].append(fix_value - sham_value)

    # PAC
    this_df = df_pac.query("Subj=='{}' and StimType=='sham'".format(subj))
    nd_sham_value = this_df["ND"].values.mean()
    this_df = df_pac.query("Subj=='{}' and StimType=='fix'".format(subj))
    nd_fix_value = this_df["ND"].values.mean()
    df_dict["PAC"].append(nd_fix_value - nd_sham_value)

    # KL Measures
    this_df = df_kl.query("Subj=='{}' and Cond=='sham'".format(subj))
    div_gauss_sham = this_df["gauss_div"].values.mean()
    div_nestso_sham = this_df["SO_div"].values.mean()
    div_nestdo_sham = this_df["deltO_div"].values.mean()
    this_df = df_kl.query("Subj=='{}' and Cond=='fix'".format(subj))
    div_gauss_fix = this_df["gauss_div"].values.mean()
    div_nestso_fix = this_df["SO_div"].values.mean()
    div_nestdo_fix = this_df["deltO_div"].values.mean()
    df_dict["KLDiv_gauss"].append(div_gauss_fix - div_gauss_sham)
    df_dict["KLDiv_NestSO"].append(div_nestso_fix - div_nestso_sham)
    df_dict["KLDiv_NestDO"].append(div_nestdo_fix - div_nestdo_sham)

df = pd.DataFrame.from_dict(df_dict)
df.to_csv("../indiv_data/indiv_measures.csv")

# normalize the columns
df_norm = df.copy()
for col in df.keys():
    if col == "Subj":
        continue
    df_norm[col] = (df_norm[col]-df_norm[col].mean())/df_norm[col].std()

g = sns.PairGrid(df_norm)
g.map(sns.scatterplot)
