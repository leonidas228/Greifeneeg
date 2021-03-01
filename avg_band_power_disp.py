from os import listdir
from os.path import isdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import matplotlib
font = {'weight' : 'bold',
        'size'   : 26}
matplotlib.rc('font', **font)
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
img_dir = "../images/"

osc_freqs = {"SO (0.5-1.2Hz)":(0.5, 1.25), "deltO (0.75-4.25Hz)":(0.75, 4.25),
             "13-17Hz":(13,17)}

df = pd.read_pickle("{}avg_band_power.pickle".format(proc_dir))

for osc in osc_freqs.keys():
    this_df = df.query("OscType=='{}'".format(osc))
    fig, ax = plt.subplots(1, 1, figsize=(38.4,21.6))
    sns.barplot(data=this_df, x="StimType", y="Power", hue="Dur",
            order=["sham","eig","fix"], hue_order=["30s","2m","5m"], ax=ax)
    plt.ylabel("Power (db)")
    plt.title(osc)
    #plt.tight_layout()
    plt.savefig("{}{}.png".format(img_dir, osc))

    facet = sns.catplot(data=this_df, x="StimType", y="Power", hue="Dur",
                        col="Sync", order=["sham","eig","fix"],
                        hue_order=["30s","2m","5m"], kind="bar")
    plt.suptitle(osc)
    facet.fig.set_size_inches(38.4,21.6)
    #plt.tight_layout()
    plt.savefig("{}{}_bysync.png".format(img_dir, osc))
