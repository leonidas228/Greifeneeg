import mne
from os import listdir, getcwd
from os.path import isdir, join
import re

# key for subject, conditions/filenames
subj_key = {
    '1001': {'anodal': 'T1', 'sham': 'T3', 'cathodal': 'T4(2)'}, 
    '1002': {'sham': 'T1', 'anodal': 'T3', 'cathodal': 'T4'},
    '1003': {'anodal': 'T1', 'sham': 'T3', 'cathodal': 'T4.2'}, 
    '1004': {'sham': 'T2', 'anodal': 'T3', 'cathodal': 'T4'}, 
    '1005': {'sham': 'T1', 'anodal': 'T2', 'cathodal': 'T4'}, 
    '1006': {'sham': 'T1', 'anodal': 'T3', 'cathodal': 'T4'}, 
    '1008': {'sham': 'T1', 'anodal': 'T2', 'cathodal': 'T4'}, 
    '1011': {'sham': 'T1', 'anodal': 'T2', 'cathodal': 't4'}, 
    '1012': {'sham': 'T2', 'anodal': ['T3', 't3 weiter'], 'cathodal': 'T4'}, 
    '1013': {'anodal': 'T1', 'sham': 'T3', 'cathodal': 'T4'}, 
    '1020': {'sham': 'T1', 'anodal': ['T2', 'T2 weiter'], 'cathodal': 'T4'}, 
    '1021': {'anodal': 'T1', 'sham': 'T2', 'cathodal': 'T4'}, 
    '1023': {'anodal': 'T1', 'sham': 'T2', 'cathodal': 'T4'}, 
    '1026': {'sham': ['T1', 'T1_2'], 'anodal': ['T3', 'T3_weiter', 'T3_weiter2'], 'cathodal': 'T4'}, 
    '1036': {'sham': 'T1', 'anodal': 'T3', 'cathodal':['T4(2)', 'T4(2)_2']}, 
    '1038': {'sham': ['T1', 'T1_14_56', 't1_14_58', 'T1_15_14'], 'anodal': 'T2', 'cathodal': 'T4'},
    '1042': {'sham':'T1', 'anodal':['T3', 'T3_neu', 'T3_neu2', 't3_neu3', 't3_neu4'],
             'cathodal':['T4', 'T4_neu', 'T4_neu2', 't4_neu3', 'T4_neu4']},
    '1046':{'sham':'T1', 'anodal':'T2', 'cathodal':'T4'},
    '1054':{'sham':'T1', 'anodal':'T2', 'cathodal':['T4', 'T4_1']},
    '1055':{'sham':'T1', 'anodal':'T2', 'cathodal':'T4'},
    '1057':{'sham':'T1', 'anodal':'T2', 'cathodal':'T4'},
    }
# bad channels: last few subjects not here yet! still need to do 1054 T4, 1046, and 1057
subj_badchans = {
                "1001":{"T4(2)":["FC2", "Cz"]},
                "1002":{"T1":["Fz"], "T4":["FC2"]},
                "1006":{"T4":["FC2"]},
                "1011":{"T4":["CP1"]},
                "1013":{"T3":["Cz", "CP1"]},
                "1020":{"T1":["FC2", "Fz", "CP2"], "T4":["FC2"]},
                "1021":{"T1":["FC2", "CP1", "CP2", "Cz"], "T2":["Fz", "FC1","FC2", "Cz","CP1","CP2"], 
                        "T4":["FC1", "FC2"]},
                "1023":{"T4":["FC1", "Cz"]},
                "1026":{"T1_2":["CP2"], "T3_weiter2":["Fz", "CP2"], "T4":["FC1"]},
                "1038":{"T4":["Cz"]},
                "1042":{"T3":["Fz", "Cz"]},
                "1057":{"T2":['FC3', 'FC6', 'F8', 'FC2'], "T4":['FC3', 'FC6', 'F8', 'FC2']},
                "1054":{"T1":['FC1', 'C3', 'Pz', 'Oz', 'F1', 'CP3', 'P1', 'POz', 'C2', 'Iz', 'P2', 'P5', 'CP1', 'FC2'],
                        "T2":['C1', 'P1', 'Pz', 'Cz', 'P3', 'CPz'],
                        }
                        
}


root_dir = getcwd()

raw_dir = join(root_dir, "data/raw") # get raw files from here
proc_dir = join(root_dir, "data/proc") # save the processed files here
filelist = listdir(raw_dir) # get list of all files in raw directory
proclist = listdir(proc_dir) # and in proc directory
overwrite = True

#convert
for dirname in filelist: # cycle through all files in raw directory
    match = re.search("NAP_(\d{4})", dirname)
    if match:
        subj = match.groups()[0]
    if not match or subj not in list(subj_key.keys()):
        continue
    this_dir = join(raw_dir, dirname)
    for k, v in subj_key[subj].items():
        outfile =  f"NAP_{subj}_{k}-raw.fif"
        if outfile in proclist and not overwrite:
            print(f"{outfile} already exists. Skipping...")
            continue
        raws = []
        # for merging, uses list of Raws; in the case there is no merging, make a list of length one
        if isinstance(v, str):
            v = [v]
        for vv in v:
            filepath = join(this_dir, f"NAP_{subj}_{vv}.vhdr")
            raw = mne.io.read_raw_brainvision(filepath) # convert
            if subj in list(subj_badchans.keys()):
                if vv in subj_badchans[subj]:
                    raw.info["bads"] = subj_badchans[subj][vv]
            raws.append(raw)
        # if a channel is bad in one (sub)recording, mark it bad in all of them
        all_bads = [c for r in raws for c in r.info["bads"]]
        for r in raws:
            r.info["bads"] = all_bads

        raw = mne.concatenate_raws(raws)
        raw.save(join(proc_dir, outfile), overwrite=overwrite)

# # sfb 1 sham conditions
# subj_key = {
#             "1001":"043_T2",
#             "1002":"053_T2",
#             "1003":"044_T4",
#             "1004":"003_T8",
#             "1005":"002_T2",
#             "1006":"022_T3",
#             "1008":"026_T6",
#             "1011":"046_T7",
#             "1012":"050_T8",
#             "1013":"037_T7",
#             "1023":"025_T2"
# }
# subj_badchans = {
#     "1001":["Cz"],
#     "1002":["FC1"],
#     "1012":["FC1"]
# }

# sfb1_dir = "/home/jev/hdd/sfb/raw/"
# for k, v in subj_key.items():
#     raws = []
#     # for merging, uses list of Raws; in the case there is no merging, make a list of length one
#     if isinstance(v, str):
#         v = [v]
#     for vv in v:
#         filepath = join(sfb1_dir, f"NAP_{vv}.vhdr")
#         raw = mne.io.read_raw_brainvision(filepath) # convert
#         raws.append(raw)
#     raw = mne.concatenate_raws(raws)
#     if k in list(subj_badchans.keys()):
#         raw.info["bads"] = subj_badchans[k]
#     raw.save(join(proc_dir, f"NAP_{k}_sfb1-raw.fif"),
#             overwrite=overwrite)
