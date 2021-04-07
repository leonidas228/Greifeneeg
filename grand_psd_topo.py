import mne
from mne.time_frequency import psd_array_multitaper, psd_multitaper, psd_welch, psd_array_welch
from mne.viz.utils import _convert_psds
from mne.viz.topomap import _find_topomap_coords, _plot_topomap_multi_cbar
from os import listdir
from os.path import isdir
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)
from scipy.integrate import simps

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)
overwrite = False # skip
dB = False

freq_ranges = {"SO":(0.5,1.25),"DO":(0.75, 4.25),"low_spind":(12,15),
               "high_spind":(16,20)}
vms = [(1, .7), (1.5, -1)]
freq_res = 0.1

fmax = 25
conds = ["sham", "eig", "fix"]
#conds = ["jung", "alt", "mci"]
inst_type = "epo"
cond_psds = {cond:{fr:[] for fr in freq_ranges.keys()} for cond in conds}
for cond in conds:
    for filename in filelist:
        this_match = re.match("ibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
        if this_match:
            subj, cond = this_match.group(1), this_match.group(2)
            inst = mne.io.Raw(proc_dir+filename, preload=True)
            inst.interpolate_bads()

            if "eig" in cond:
                cond = "eig"
            elif "fix" in cond:
                cond = "fix"
            elif "sham" in cond:
                cond = "sham"

            inst.set_montage("standard_1020")
            picks = mne.pick_types(inst.info, eeg=True)
            pos = _find_topomap_coords(inst.info, picks=picks)
            psd, freqs = psd_multitaper(inst, fmax=25, n_jobs=1)

            for freq_name, freq_range in freq_ranges.items():
                freq_inds = np.logical_and(freqs >= freq_range[0],
                                           freqs <= freq_range[1])
                log_psd = np.log10(psd[...,freq_inds])
                power_spec_topo = simps(log_psd, dx=freqs[1]-freqs[0],
                                        axis=-1)
                cond_psds[cond][freq_name].append(power_spec_topo)


    ylabel = _convert_psds(psd, False, "power", 1e+6, 'µV')
    cond_psds["pos"] = pos
    cond_psds["ylabel"] = ylabel

    with open("{}cond_psds.pickle".format(proc_dir), "wb") as f:
        pickle.dump(cond_psds, f)

    # _plot_topomap_multi_cbar(power_spec_topo, pos, axes, colorbar=True,
    #                          unit="log($\\mathrm"+ylabel[8:-5]+"}$)",
    #                          vmin=vm[0], vmax=vm[1])
    #plt.savefig("{}_{}-{}Hz.svg".format(cond_title[cond], freq_range[0], freq_range[1]), format="svg")
