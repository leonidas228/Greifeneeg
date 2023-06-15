import mne
from os import listdir
from os.path import isdir
import re
import numpy as np


root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)

conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham30s", "sham2m", "sham5m"]
excludes = []

chan_dict = {"Vo":"eog","Vu":"eog","Re":"eog","Li":"eog","MovRe":"misc",
             "MovLi":"misc"}
orig_chans = list(chan_dict.keys())
with open("nonEEGchans.txt","wt") as f:
    f.write("Subject\tTag")
    for k in chan_dict.keys():
        f.write("\t"+k)
    f.write("\tEEG\n")

for filename in filelist:
    this_match = re.match("caf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if (cond not in conds) or ("{}_{}".format(subj,cond) in excludes):
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)

        # some sessions name channels differently
        mov_picks = mne.pick_channels(raw.ch_names, include=["Mov re", "Mov li"])
        if len(mov_picks):
            ml, mr = "Mov li", "Mov re"
        else:
            ml, mr = "MovLi", "MovRe"
        chan_dict = {"Vo":"eog","Vu":"eog","Re":"eog","Li":"eog",mr:"misc",
                     ml:"misc"}
        orig_chans = list(chan_dict.keys())

        subj, tag = this_match.group(1), this_match.group(2)
        with open("nonEEGchans.txt","at") as f:
            f.write("{}\t{}".format(subj,tag))
            for k,v in chan_dict.items():
                try:
                    raw.set_channel_types({k:v})
                    f.write("\t1")
                except:
                    f.write("\t0")
            for ch_idx, ch in enumerate(raw.get_channel_types()):
                if ch == "eeg":
                    f.write("\t"+raw.ch_names[ch_idx])
            f.write("\n")

        try:
            data = np.empty((0,len(raw)))
            eog_v_picks = mne.pick_channels(raw.ch_names, include=["Vo", "Vu"])
            temp_data = raw.get_data()[eog_v_picks,]
            data = np.vstack((data, temp_data[0,] - temp_data[1,]))
            eog_h_picks = mne.pick_channels(raw.ch_names, include=["Re", "Li"])
            temp_data = raw.get_data()[eog_h_picks,]
            data = np.vstack((data, temp_data[0,] - temp_data[1,]))
            mov_picks = mne.pick_channels(raw.ch_names, include=[ml, mr])
            temp_data = raw.get_data()[mov_picks,]
            data = np.vstack((data, temp_data[0,] - temp_data[1,]))
            info = mne.create_info(["eog_v","eog_h","mov"], sfreq=raw.info["sfreq"],
                                   ch_types=["eog","eog","misc"])
            non_eeg = mne.io.RawArray(data, info)
            raw.add_channels([non_eeg], force_update_info=True)
            raw.drop_channels(orig_chans)
            raw.save("{}scaf_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond),
                         overwrite=True)
        except:
            pass
