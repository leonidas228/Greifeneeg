from os.path import isdir
from os import listdir
import re
import numpy as np
import mne
import numpy as np
import pickle
from numpy.random import shuffle
from mne.time_frequency import tfr_array_morlet
import pandas as pd
import matplotlib.pyplot as plt

# replace numpy.bincount, which can't handle bins without values
def bincount(x, bin_n):
    if len(x.shape) == 1:
        x = np.expand_dims(x,1)
    elif len(x.shape) > 2:
        raise ValueError("Too many dimensions in input x")
    bincounts = np.zeros((bin_n, x.shape[1]))
    for b_idx in range(bin_n):
        bincounts[b_idx,] = (x==b_idx).sum(axis=0)
    return bincounts

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

perm_n = 1000
overwrite = True

filelist = listdir(proc_dir)

overlaps = 0
total_osc = 0
for filename in filelist:
    # load files, merge SO and deltO annotations
    this_match = re.match("NAP_(.*)_(.*)_(.*)_SO-raw.fif", filename)
    if not this_match:
        continue
    subj, cond, chan = (this_match.group(1), this_match.group(2),
                        this_match.group(3))
    outfile = "spindle_distros_{}_{}_{}.pickle".format(subj, cond,
                                                       chan)
    if outfile in filelist and not overwrite:
        print("{} already exists, skipping.".format(outfile))
        continue

    raw = mne.io.Raw(proc_dir+filename)
    deltO_filename = "NAP_{}_{}_{}_deltO-raw.fif".format(subj, cond, chan)
    raw_deltO = mne.io.Raw(proc_dir+deltO_filename)
    annot = raw.annotations
    annot = annot.__add__(raw_deltO.annotations)
    raw.set_annotations(annot)
    if len(raw.annotations) < 50:
        print("{} has too few oscillations, skipping...".format(outfile))
        continue

    # mark areas in the raw where which oscillations occurred, if any
    event_bools = np.zeros((3, len(raw)), dtype=bool)
    for annot in raw.annotations.__iter__():
        an_name = annot["description"]
        if "Trough" in an_name or "Peak" in an_name:
            continue
        if "deltO" in an_name:
            bool_idx = 1
        elif "SO" in an_name:
            bool_idx = 0
        else:
            raise ValueError("Event {} cannot be identified.".format(an_name))
        onset_idx = raw.time_as_index(annot["onset"] - raw.first_time)[0]
        offset_idx = raw.time_as_index(annot["onset"] + annot["duration"] -
                                       raw.first_time)[0]

        event_bools[bool_idx, onset_idx:offset_idx] = 1

    event_bools[2,] = ~(event_bools[0,] | event_bools[1,])
    min_count = np.min(event_bools.sum(axis=1))

    overlaps += len(np.where(event_bools[0,] & event_bools[1,])[0])
    total_osc += len(np.where(event_bools[0,] | event_bools[1,])[0])

    # calculate tfr of raw
    raw.pick_channels(["central"])
    data = np.expand_dims(raw.get_data() * 1e+6, 0)

    tfr = tfr_array_morlet(data, raw.info["sfreq"], np.arange(12,18),
                           output="power")
    tfr = tfr.squeeze().mean(axis=0)
    tfr = np.log10(tfr)


    boot_num = min_count - 5

    SO_tfr = tfr[event_bools[0,]]
    deltO_tfr = tfr[event_bools[1,]]
    free_tfr = tfr[event_bools[2,]]

    all_bin_edges = []
    all_bin_counts = []
    for perm_idx in range(perm_n):
        # sample the three types
        shuffle(SO_tfr)
        shuffle(deltO_tfr)
        shuffle(free_tfr)
        SO_samp = SO_tfr[:boot_num]
        deltO_samp = deltO_tfr[:boot_num]
        free_samp = free_tfr[:boot_num]

        # derive general histogram
        hist_array = np.vstack([SO_samp, deltO_samp, free_samp]).T
        bin_edges = np.histogram_bin_edges(hist_array.flatten(), bins=100)
        all_bin_edges.append(bin_edges)

        # see in which bins each type tended to land
        digits = np.digitize(hist_array, bin_edges)
        bincounts = bincount(digits, len(bin_edges))
        all_bin_counts.append(bincounts)

    # average, wrap up in a dictionary, save
    hist_dict = {}
    avg_bin_edges = np.array(all_bin_edges).mean(axis=0)
    avg_bin_counts = np.array(all_bin_counts).mean(axis=0) / boot_num
    hist_dict["bin_edges"] = avg_bin_edges
    hist_dict["SO_counts"] = avg_bin_counts[:,0]
    hist_dict["deltO_counts"] = avg_bin_counts[:,1]
    hist_dict["free_counts"] = avg_bin_counts[:,2]
    hist_dict["boot_num"] = boot_num

    with open(proc_dir+outfile, "wb") as f:
        pickle.dump(hist_dict, f)
