import mne
from os import listdir
import re
from os.path import join
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()

def norm_overlay(x, min=10, max=20, centre=15, xmax=4e-05):
    x = x / xmax
    x = x * (max-min)/2 + centre
    return x

def graph(ur_epo, title, ROI, vmin=-3, vmax=3):
    subj = ur_epo.metadata.iloc[0]["Subj"]
    print(subj)
    epo = ur_epo.copy()[f"ROI=='{ROI}'"]
    epo.pick_channels([ROI])
    tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                    n_jobs=n_jobs)
    tfr.crop(-2.25, 2.25)
    tfr.apply_baseline((-2.25, -1), mode="zscore")

    stim_an_n = len(epo["Cond=='stim' and Polarity=='anodal'"])
    stim_ca_n = len(epo["Cond=='stim' and Polarity=='cathodal'"])
    sham_an_n = len(epo["Cond=='sham' and Polarity=='anodal'"])
    sham_ca_n = len(epo["Cond=='sham' and Polarity=='cathodal'"])

    # get SO for overlays
    epo.crop(-2.25, 2.25)
    so_stim_an = norm_overlay(epo["Cond=='stim' and Polarity=='anodal'"].average().data.squeeze())
    so_stim_ca = norm_overlay(epo["Cond=='stim' and Polarity=='cathodal'"].average().data.squeeze())
    so_sham_an = norm_overlay(epo["Cond=='sham' and Polarity=='anodal'"].average().data.squeeze())
    if sham_ca_n:
        so_sham_ca = norm_overlay(epo["Cond=='sham' and Polarity=='cathodal'"].average().data.squeeze())

    line_color = "white"
    # define layout of graph
    mos_str = """
            AABBCC
            AABBCC
            DDEEFF
            DDEEFF
            """
    fig, axes = plt.subplot_mosaic(mos_str, figsize=(38.4, 21.6))
    tfr_stim_an = tfr["Cond=='stim' and Polarity=='anodal'"].average()
    tfr_stim_ca = tfr["Cond=='stim' and Polarity=='cathodal'"].average()
    tfr_sham_an = tfr["Cond=='sham' and Polarity=='anodal'"].average()
    if sham_ca_n:
        tfr_sham_ca = tfr["Cond=='sham' and Polarity=='cathodal'"].average()

    tfr_stim_an.plot(vmin=0, vmax=vmax, axes=axes["A"], cmap="hot")
    axes["A"].set_title(f"Stimulation Anodal ({stim_an_n})")
    axes["A"].plot(tfr_stim_an.times, so_stim_an, color=line_color)

    tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes["B"], cmap="hot")
    axes["B"].set_title(f"Sham Anodal ({sham_an_n})")
    axes["B"].plot(tfr_sham_an.times, so_sham_an, color=line_color)

    (tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["C"])
    axes["C"].set_title("Stim - Sham Anodal")
    axes["C"].plot(tfr_stim_an.times, so_stim_an, color="black")

    tfr_stim_ca.plot(vmin=0, vmax=vmax, axes=axes["D"], cmap="hot")
    axes["D"].set_title(f"Stimulation Cathodal ({stim_ca_n})")
    axes["D"].plot(tfr_stim_ca.times, so_stim_ca, color=line_color)

    if sham_ca_n:
        tfr_sham_ca.plot(vmin=0, vmax=vmax, axes=axes["E"], cmap="hot")
        axes["E"].set_title(f"Sham Cathodal ({sham_ca_n})")
        axes["E"].plot(tfr_sham_an.times, so_sham_ca, color=line_color)

        (tfr_stim_ca-tfr_sham_ca).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["F"])
        axes["F"].set_title("Stim - Sham Cathodal")
        axes["F"].plot(tfr_stim_ca.times, so_stim_ca, color="black")

    plt.suptitle(title)
    plt.savefig(join(fig_dir, f"{title}.png"))
    plt.close()

def graph_subjavg(ur_epo, title, ROI, vmin=-3, vmax=3):
    epo = ur_epo.copy()[f"ROI=='{ROI}'"]
    epo.pick_channels([ROI])
    tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                    n_jobs=n_jobs)
    tfr.crop(-2.25, 2.25)
    tfr.apply_baseline((-2.25, -1), mode="zscore")
    epo.crop(-2.25, 2.25)

    subjs = list(ur_epo.metadata["Subj"].unique())
    so_stim_an, so_stim_ca, so_sham_an, so_sham_ca = [], [], [], []
    tfr_stim_an, tfr_stim_ca, tfr_sham_an, tfr_sham_ca = [], [], [], []
    for subj in subjs:
        subj_epo = epo.copy()[f"Subj=='{subj}'"]
        tfr_epo = tfr.copy()[f"Subj=='{subj}'"]
        if 0 in (
            len(subj_epo.copy()["Cond=='stim' and Polarity=='anodal'"]),  
            len(subj_epo.copy()["Cond=='stim' and Polarity=='cathodal'"]),
            len(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"])
            ):
            print(f"crap {subj}")
            continue

        # get SO for overlays
        so_stim_an.append(subj_epo.copy()["Cond=='stim' and Polarity=='anodal'"].average())
        so_stim_ca.append(subj_epo.copy()["Cond=='stim' and Polarity=='cathodal'"].average())
        so_sham_an.append(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"].average())
        if len(subj_epo.copy()["Cond=='sham' and Polarity=='cathodal'"]):
            so_sham_ca.append(subj_epo.copy()["Cond=='sham' and Polarity=='cathodal'"].average())

        tfr_stim_an.append(tfr_epo.copy()["Cond=='stim' and Polarity=='anodal'"].average())
        tfr_stim_ca.append(tfr_epo.copy()["Cond=='stim' and Polarity=='cathodal'"].average())
        tfr_sham_an.append(tfr_epo.copy()["Cond=='sham' and Polarity=='anodal'"].average())
        if len(subj_epo.copy()["Cond=='sham' and Polarity=='cathodal'"]):
            tfr_sham_ca.append(tfr_epo.copy()["Cond=='sham' and Polarity=='cathodal'"].average())

    so_stim_an = norm_overlay(mne.grand_average(so_stim_an).data.squeeze())
    so_stim_ca = norm_overlay(mne.grand_average(so_stim_ca).data.squeeze())
    so_sham_an = norm_overlay(mne.grand_average(so_sham_an).data.squeeze())
    so_sham_ca = norm_overlay(mne.grand_average(so_sham_ca).data.squeeze())

    stim_an_n = len(tfr_stim_an)
    stim_ca_n = len(tfr_stim_ca)
    sham_an_n = len(tfr_sham_an)
    sham_ca_n = len(tfr_sham_ca)

    tfr_stim_an = mne.grand_average(tfr_stim_an)
    tfr_stim_ca = mne.grand_average(tfr_stim_ca)
    tfr_sham_an = mne.grand_average(tfr_sham_an)
    tfr_sham_ca = mne.grand_average(tfr_sham_ca)


    line_color = "white"
    # define layout of graph
    mos_str = """
            AABBCC
            AABBCC
            DDEEFF
            DDEEFF
            """
    fig, axes = plt.subplot_mosaic(mos_str, figsize=(38.4, 21.6))

    tfr_stim_an.plot(vmin=0, vmax=vmax, axes=axes["A"], cmap="hot")
    axes["A"].set_title(f"Stimulation Anodal ({stim_an_n})")
    axes["A"].plot(tfr_stim_an.times, so_stim_an, color=line_color)

    tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes["B"], cmap="hot")
    axes["B"].set_title(f"Sham Anodal ({sham_an_n})")
    axes["B"].plot(tfr_sham_an.times, so_sham_an, color=line_color)

    (tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["C"])
    axes["C"].set_title("Stim - Sham Anodal")
    axes["C"].plot(tfr_stim_an.times, so_stim_an, color="black")

    tfr_stim_ca.plot(vmin=0, vmax=vmax, axes=axes["D"], cmap="hot")
    axes["D"].set_title(f"Stimulation Cathodal ({stim_ca_n})")
    axes["D"].plot(tfr_stim_ca.times, so_stim_ca, color=line_color)

    tfr_sham_ca.plot(vmin=0, vmax=vmax, axes=axes["E"], cmap="hot")
    axes["E"].set_title(f"Sham Cathodal ({sham_ca_n})")
    axes["E"].plot(tfr_sham_ca.times, so_sham_ca, color=line_color)

    (tfr_stim_ca-tfr_sham_ca).plot(vmin=vmin/2, vmax=vmax/2, axes=axes["F"])
    axes["F"].set_title("Stim - Sham Cathodal")
    axes["F"].plot(tfr_stim_ca.times, so_stim_ca, color="black")

    plt.suptitle(title)
    plt.savefig(join(fig_dir, f"{title}.png"))
    plt.close()


root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")
fig_dir = join(root_dir, "figs")

freqs = np.linspace(10, 20, 50)
n_cycles = 5
n_jobs = 24

ur_epo = mne.read_epochs(join(proc_dir, f"grand-epo.fif"))
ur_epo = ur_epo["OscType=='SO'"]
subjs = list(ur_epo.metadata["Subj"].unique())
ROIs = list(ur_epo.metadata["ROI"].unique())
for ROI in ROIs:
    # graph(ur_epo, f"all epochs {ROI}", ROI)
    # graph_subjavg(ur_epo, f"subj avg {ROI}", ROI)
    for subj in subjs:
        subj_epo = ur_epo.copy()[f"Subj=='{subj}'"]
        graph(subj_epo, f"{subj} {ROI}", ROI, vmin=-6, vmax=6)


