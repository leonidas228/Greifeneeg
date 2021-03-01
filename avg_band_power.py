import mne
from os import listdir
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import simps
from os.path import isdir
from mne.time_frequency import psd_multitaper
plt.ion()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

filelist = listdir(proc_dir)

chan_groups = {"central":["Fz","FC1","FC2", "Cz","CP1","CP2"]}
osc_freqs = {"SO (0.5-1.2Hz)":(0.5, 1.25), "deltO (0.75-4.25Hz)":(0.75, 4.25),
             "12-15Hz":(12,15), "13-17Hz":(13,17), "15-20Hz":(15,20)}
bandwidth = 0.2
df_dict = {"Subj":[], "Sync":[], "Cond":[], "StimType":[], "Dur":[],
           "OscType":[], "Rel_Power":[], "Power":[], "Index":[]}
n_jobs = 8

for filename in filelist:
    this_match = re.match("ibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if not this_match:
        continue
    subj, cond = this_match.group(1), this_match.group(2)

    raw = mne.io.Raw(proc_dir+filename,preload=True)

    # begin at first post stim period
    for annot in raw.annotations:
        if annot["description"] == "Post_Stimulation 0":
            break
    raw.crop(tmin=(annot["onset"] - raw.first_samp / raw.info["sfreq"]))
    # cycle through post-stim periods
    for annot in raw.annotations:
        this_match = re.match("Post_Stimulation (\d)", annot["description"])
        if not this_match:
            continue
        if annot["duration"] < 50:
            continue
        stim_idx = this_match.group(1)
        onset = annot["onset"] - raw.first_samp / raw.info["sfreq"]
        endpoint = onset + annot["duration"]
        onset += 10 # add 10 to be extra sure stim artefacts don't come in
        psd, freqs = psd_multitaper(raw, tmin=onset, tmax=endpoint,
                                    picks=chan_groups["central"], fmax=100,
                                    bandwidth=bandwidth, adaptive=True,
                                    n_jobs=n_jobs)
        psd = 10. * np.log10(psd* 1e+12) #dB
        #psd = np.log10(psd * 1e+12) # log of mV
        psd = psd.mean(axis=0)
        freq_res = freqs[1] - freqs[0]
        total_power = simps(psd, dx=freq_res)
        for of_k, of_v in osc_freqs.items():
            freq_min_idx = find_nearest(freqs, of_v[0])
            freq_max_idx = find_nearest(freqs, of_v[1])
            power = simps(psd[freq_min_idx:freq_max_idx], dx=freq_res)
            rel_power = power / total_power

            df_dict["Subj"].append(subj)
            if int(subj) < 31 and subj != "021":
                df_dict["Sync"].append("async")
            else:
                df_dict["Sync"].append("sync")

            df_dict["Cond"].append(cond)
            if "eig" in cond:
                df_dict["StimType"].append("eig")
            elif "fix" in cond:
                df_dict["StimType"].append("fix")
            elif "sham" in cond:
                df_dict["StimType"].append("sham")
            if "30s" in cond:
                df_dict["Dur"].append("30s")
            elif "2m" in cond:
                df_dict["Dur"].append("2m")
            elif "5m" in cond:
                df_dict["Dur"].append("5m")

            df_dict["OscType"].append(of_k)
            df_dict["Rel_Power"].append(rel_power)
            df_dict["Power"].append(power)
            df_dict["Index"].append(stim_idx)

df = pd.DataFrame.from_dict(df_dict)
df.to_pickle("{}avg_band_power.pickle".format(proc_dir))
