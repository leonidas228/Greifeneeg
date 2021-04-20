import mne
from os import listdir
from os.path import isdir
import re
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import pandas as pd
plt.ion()

def check_trough_annot(desc):
    event = None
    if "Trough" in desc:
        event = 0
        if "posterior" in desc:
            event += 200
        if "deltO" in desc:
            event += 100
        if "Post" in desc:
            event += 50
        event += int(desc[-1])
    return event

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)
channel = "central"
spindle_freq = np.arange(11,14)
n_jobs = 4
gw_time = 0.25

for filename in filelist:
    this_match = re.match("aibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        raw_work = raw.copy()
        ft = raw_work.first_time
        raw_work.pick_channels([channel])
        epo = mne.make_fixed_length_epochs(raw_work, duration=raw_work.times[-1])
        power = tfr_morlet(epo, spindle_freq, n_cycles=5, average=False,
                           return_itc=False, n_jobs=n_jobs)
        tfr = np.mean(power.data[0,],axis=1)
        gw_len = np.round(gw_time * raw.info["sfreq"]).astype(int)
        gauss_win = np.exp(-0.5*((np.arange(gw_len)-gw_len/2)/(0.5*gw_len/2))**2)
        #tfr_con = np.convolve(tfr, gauss_win, mode="same")
        tfr_con = tfr.copy()
        tfr_aschan = np.zeros((1,len(raw_work)))
        tfr_aschan[0,:tfr_con.shape[1]] = tfr_con
        tfr_raw = mne.io.RawArray(tfr_aschan, mne.create_info(["TFR"],raw_work.info["sfreq"],ch_types="eeg"))
        raw_work.add_channels([tfr_raw], force_update_info=True)

        # epoching and database
        events = mne.events_from_annotations(raw, check_trough_annot)
        df_dict = {"Subj":[],"Cond":[],"PrePost":[],"Ort":[],"OscType":[],
                   "Index":[],"Stim":[]}
        for event in np.nditer(events[0][:,-1]):
            eve = event.copy()
            if eve >= 200:
                df_dict["Ort"].append("posterior")
                eve -= 200
            else:
                df_dict["Ort"].append("central")
            if eve >= 100:
                df_dict["OscType"].append("deltO")
                eve -= 100
            else:
                df_dict["OscType"].append("SO")
            if eve >= 50:
                df_dict["PrePost"].append("Post")
                eve -= 50
            else:
                df_dict["PrePost"].append("Pre")
            df_dict["Index"].append(int(eve))
            df_dict["Subj"] = subj
            df_dict["Cond"] = cond
            if cond != "sham":
                df_dict["Stim"].append("stim")
            else:
                df_dict["Stim"].append("sham")
        df = pd.DataFrame.from_dict(df_dict)
        epo = mne.Epochs(raw_work, events[0], tmin=-1.25, tmax=1.25, detrend=None,
                         baseline=(-1.25,-0.75), metadata=df, event_repeated="drop").load_data()
        epo.save("{}spin_NAP_{}_{}-epo.fif".format(proc_dir,subj,cond), overwrite=True)
