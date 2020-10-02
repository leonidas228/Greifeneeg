import mne
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

oscs = ["SO", "deltO"]
conds = ["eig30s", "fix30s"]

epo = mne.read_epochs("{}grand-epo.fif".format(proc_dir), preload=True)
df = epo.metadata
sub_inds = df["Subj"].values.astype(int) >= 31

epo.filter(l_freq=0.3,h_freq=3,n_jobs=8)

# e = epo["Cond=='{}' or Cond=='{}'".format(conds[0],conds[1])]
# evos = []
# for osc in oscs:
#     this_e = e["OscType=='{}' and (Cond=='eig30s' or Cond=='fix30s')".format(osc)]
#     this_e.plot_image(picks="central")
#     plt.suptitle("30 second stimulation, {}".format(osc))
#     evos.append(this_e.average())
#     evos[-1].comment = osc
# mne.viz.plot_compare_evokeds(evos,picks="central")

colors, styles = [], []
for col in ["blue", "red", "green", "cyan", "purple"]:
    for sty in ["dotted", "solid"]:
        colors.append(col)
        styles.append(sty)

for osc in ["SO"]:
    epo0 = epo["OscType=='{}'".format(osc)]
    for cond in ["eig", "fix"]:
        for timelen in ["30s", "2m", "5m"]:
            epo1 = epo0["Cond=='{}{}'".format(cond,timelen)]
            evo_inds = []
            for ind in range(5):
                epo2 = epo1["Index=='{}'".format(ind)]
                for pp in ["Pre", "Post"]:
                    epo3 = epo2["PrePost=='{}'".format(pp)]
                    evo = epo3.average()
                    evo.comment = "{} {}{} {} {}".format(osc, cond, timelen,
                                                         ind, pp)
                    evo_inds.append(evo)
            mne.viz.plot_compare_evokeds(evo_inds, picks="central",
                                         colors=colors, linestyles=styles,
                                         ylim={"eeg":(-40,20)})
