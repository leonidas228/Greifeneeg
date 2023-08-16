import mne
from os import listdir
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
             'cathodal':['T4', 'T4_neu', 'T4_neu2', 't4_neu3', 'T4_neu4']}
    }

root_dir = "/home/jev/hdd/sfb2/"

raw_dir = join(root_dir, "raw") # get raw files from here
proc_dir = join(root_dir, "proc") # save the processed files here
filelist = listdir(raw_dir) # get list of all files in raw directory
proclist = listdir(proc_dir) # and in proc directory
overwrite = True # skip
do_subj = None

# convert
for dirname in filelist: # cycle through all files in raw directory
    match = re.search("NAP_(\d{4})", dirname)
    if match:
        subj = match.groups()[0]
    if not match or subj not in list(subj_key.keys()):
        continue
    this_dir = join(raw_dir, dirname)
    for k, v in subj_key[subj].items():
        raws = []
        # for merging, uses list of Raws; in the case there is no merging, make a list of length one
        if isinstance(v, str):
            v = [v]
        for vv in v:
            filepath = join(this_dir, f"NAP_{subj}_{vv}.vhdr")
            raw = mne.io.read_raw_brainvision(filepath) # convert
            raws.append(raw)
        raw = mne.concatenate_raws(raws)
        raw.save(join(proc_dir, f"NAP_{subj}_{k}-raw.fif"),
                overwrite=overwrite)

