import mne
from os import listdir
import re
from os.path import isdir
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.ion()

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","eig2m","eig30s","sham30s","fix5m","fix2m","fix30s"]
cond_names = ["Eigen 5m", "Eigen 2m", "Eigen 30s", "Sham", "Fixed 5m", "Fixed 2m", "Fixed 30s"]
subj = "043"
chans = ["Fz", "FC1", "FC2", "Cz", "CP1", "CP2"]
disp_len = 3500
outcut = (810000, 840000)
ylim = (-0.01, 0.01)

fig, axes = plt.subplots(len(conds)+1, 1, figsize=(38.4, 21.6))
for cond_idx, (cond, cond_name) in enumerate(zip(conds, cond_names)):
    filename = "af_NAP_{}_{}-raw.fif".format(subj, cond)
    raw = mne.io.Raw(proc_dir+filename,preload=True)
    for annot in raw.annotations:
        if annot["description"] == "BAD_Stimulation 0":
            start_point = annot["onset"]
        elif annot["description"] == "BAD_Stimulation 4":
            end_point = annot["onset"] + annot["duration"]
    remain_time = disp_len - (end_point - start_point)
    disp_start = start_point - remain_time / 2
    disp_end = end_point + remain_time / 2
    disp_start = 0 if disp_start < 0 else disp_start
    raw.crop(tmin=disp_start, tmax=disp_end)
    raw_pick = raw.copy().pick_channels(chans)
    avg_signal = raw_pick.get_data().mean(axis=0, keepdims=True)
    axes[cond_idx].plot(avg_signal.T, color="black")
    axes[cond_idx].set_ylim(ylim)
    axes[cond_idx].axis("off")
    axes[cond_idx].text(0, 1, cond_name, fontsize=32,
                        transform=axes[cond_idx].transAxes)

axes[-1].plot(avg_signal[0, outcut[0]:outcut[1]], color="black")
axes[-1].set_ylim(ylim)
axes[-1].axis("off")

rect = patches.Rectangle((outcut[0], ylim[0]), outcut[1]-outcut[0],
                          ylim[1]-ylim[0], edgecolor="black", facecolor="none",
                          linewidth=3)
axes[-2].add_patch(rect)


plt.tight_layout()
plt.savefig("../images/stim_schem_B.svg")
