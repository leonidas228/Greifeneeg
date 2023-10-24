
from mne.time_frequency import tfr_morlet
import numpy as np
import mne
from os.path import join
from scipy.stats import ttest_rel
from erpac import tfce_correct
import matplotlib.pyplot as plt
from mne.stats.cluster_level import _find_clusters
from joblib import Parallel, delayed

def norm_overlay(x, min=10, max=20, centre=15, xmax=4e-05):
    x = x / xmax
    x = x * (max-min)/2 + centre
    return x

def permute(perm_idx, a, b):
    print("Permutation {} of {}".format(perm_idx, perm_n))
    swap_inds = np.random.randint(0,2, size=len(a)).astype(bool)
    swap_a, swap_b  = a.copy(), b.copy()
    swap_a[swap_inds,] = b[swap_inds,]
    swap_b[swap_inds,] = a[swap_inds,]
    t = ttest_rel(swap_a, swap_b, axis=0)
    try:
        clusts = tfce_correct(t[0][0,])
    except:
        breakpoint()
    return abs(clusts).max()

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")
fig_dir = join(root_dir, "figs")
freqs = np.linspace(12, 18, 50)
n_cycles = 5
n_jobs = 24
threshold = 0.05
vmin, vmax = -3, 3
tfce_thresh = dict(start=0, step=0.2)
analy_crop = [-1, 1]

do_permute = False
perm_n = 1024

ur_epo = mne.read_epochs(join(proc_dir, f"grand-epo.fif"))
ur_epo = ur_epo["OscType=='SO'"]
subjs = list(ur_epo.metadata["Subj"].unique())
ROIs = list(ur_epo.metadata["ROI"].unique())

for ROI in ROIs:
    epo = ur_epo.copy()[f"ROI=='{ROI}'"]
    epo.pick_channels([ROI])
    tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                    n_jobs=n_jobs)
    tfr.apply_baseline((-2.25, -1), mode="zscore")
    tfr.crop(*analy_crop)
    epo.crop(*analy_crop)

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
            print("crap")
            breakpoint()
            continue

        # get SO for overlays
        so_stim_an.append(subj_epo.copy()["Cond=='stim' and Polarity=='anodal'"].average())
        so_stim_ca.append(subj_epo.copy()["Cond=='stim' and Polarity=='cathodal'"].average())
        so_sham_an.append(subj_epo.copy()["Cond=='sham' and Polarity=='anodal'"].average())

        tfr_stim_an.append(tfr_epo.copy()["Cond=='stim' and Polarity=='anodal'"].average())
        tfr_stim_ca.append(tfr_epo.copy()["Cond=='stim' and Polarity=='cathodal'"].average())
        tfr_sham_an.append(tfr_epo.copy()["Cond=='sham' and Polarity=='anodal'"].average())

    so_stim_an = norm_overlay(mne.grand_average(so_stim_an).data.squeeze())
    so_stim_ca = norm_overlay(mne.grand_average(so_stim_ca).data.squeeze())
    so_sham_an = norm_overlay(mne.grand_average(so_sham_an).data.squeeze())

    tfr_stim_an_dat = np.array([t.data for t in tfr_stim_an])
    tfr_stim_ca_dat = np.array([t.data for t in tfr_stim_ca])
    tfr_sham_an_dat = np.array([t.data for t in tfr_sham_an])
    
    an_t = ttest_rel(tfr_stim_an_dat, tfr_sham_an_dat, axis=0)
    ca_t = ttest_rel(tfr_stim_ca_dat, tfr_sham_an_dat, axis=0)

    if do_permute:
        results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(permute)(
                             i, tfr_stim_an_dat, tfr_sham_an_dat)
                             for i in range(perm_n))
        an_results = np.array(results)
        np.save(join(proc_dir, f"erpac_perm_{ROI}_an.npy"), results)
        results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(permute)(
                             i, tfr_stim_ca_dat, tfr_sham_an_dat)
                             for i in range(perm_n))
        ca_results = np.array(results)
        np.save(join(proc_dir, f"erpac_perm_{ROI}_ca.npy"), results)
    else:
        an_results = np.load(join(proc_dir, f"erpac_perm_{ROI}_an.npy"))
        ca_results = np.load(join(proc_dir, f"erpac_perm_{ROI}_ca.npy"))

    stim_an_n = len(tfr_stim_an)
    stim_ca_n = len(tfr_stim_ca)
    sham_an_n = len(tfr_sham_an)
    sham_ca_n = len(tfr_sham_ca)

    tfr_stim_an = mne.grand_average(tfr_stim_an)
    tfr_stim_ca = mne.grand_average(tfr_stim_ca)
    tfr_sham_an = mne.grand_average(tfr_sham_an)

    fig, axes = plt.subplots(2, 3, figsize=(38.4, 21.6))
    tfr_stim_an.plot(vmin=0, vmax=vmax, axes=axes[0,0], cmap="hot")
    axes[0,0].set_title(f"Stimulation Anodal ({stim_an_n})")
    axes[0,0].plot(tfr_stim_an.times, so_stim_an, color="white")

    tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes[0,1], cmap="hot")
    axes[0,1].set_title(f"Sham Anodal ({sham_an_n})")
    axes[0,1].plot(tfr_sham_an.times, so_sham_an, color="white")

    an_c = _find_clusters(an_t[0][0,], threshold=tfce_thresh)
    an_c = np.reshape(an_c[1], an_t[0].shape)
    thresh_val = np.quantile(an_results, 1-threshold/2)
    mask = abs(an_c) > thresh_val
    (tfr_stim_an-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes[0,2],
                                   mask_style="contour", mask=mask)
    axes[0,2].set_title("Stim Anodal - Sham Anodal")
    axes[0,2].plot(tfr_stim_an.times, so_stim_an, color="black")

    tfr_stim_ca.plot(vmin=0, vmax=vmax, axes=axes[1,0], cmap="hot")
    axes[1,0].set_title(f"Stimulation Cathodal ({stim_an_n})")
    axes[1,0].plot(tfr_stim_ca.times, so_stim_ca, color="white")

    tfr_sham_an.plot(vmin=0, vmax=vmax, axes=axes[1,1], cmap="hot")
    axes[1,1].set_title(f"Sham Anodal ({sham_an_n})")
    axes[1,1].plot(tfr_sham_an.times, so_sham_an, color="white")

    ca_c = _find_clusters(ca_t[0][0,], threshold=tfce_thresh)
    ca_c = np.reshape(ca_c[1], ca_t[0].shape)
    thresh_val = np.quantile(ca_results, 1-threshold/2)
    mask = abs(ca_c) > thresh_val
    (tfr_stim_ca-tfr_sham_an).plot(vmin=vmin/2, vmax=vmax/2, axes=axes[1,2],
                                    mask_style="contour", mask=mask)
    axes[1,2].set_title("Stim Cathodal - Sham Anodal")
    axes[1,2].plot(tfr_stim_an.times, so_stim_ca, color="black")

    plt.suptitle(ROI)
    plt.tight_layout()
    plt.savefig(join(fig_dir, f"ttest_{ROI}.png"))
