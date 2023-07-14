import mne
from os import listdir
from os.path import isdir, join
import re

# key for subject, conditions/filenames
subj_key = {"1001":{"T1":"anodal", "T3":"sham", "T4(2)":"cathodal"},
            "1002":{"T1":"sham", "T3":"anodal", "T4":"cathodal"},
            "1003":{"T1":"anodal", "T3":"sham", "T4.2":"cathodal"},
            "1004":{"T2":"sham", "T3":"anodal", "T4":"cathodal"},
            "1005":{"T1":"sham", "T2":"anodal", "T4":"cathodal"},
            "1006":{"T1":"sham", "T3":"anodal", "T4":"cathodal"},
            "1008":{"T1":"sham", "T2":"anodal", "T4":"cathodal"},
            "1011":{"T1":"sham", "T2":"anodal", "t4":"cathodal"},
            "1012":{"T2":"sham", "T3":"anodal", "T4":"cathodal"},
            "1013":{"T1":"anodal", "T3":"sham", "T4":"cathodal"},
            "1020":{"T1":"sham", "T2":"anodal", "T4":"cathodal"},
            "1021":{"T1":"anodal", "T2":"sham", "T4":"cathodal"},
            "1023":{"T1":"anodal", "T2":"sham", "T4":"cathodal"},
            "1026":{"T1":"sham", "T3":"anodal", "T4":"cathodal"},
            "1036":{"T1":"sham", "T3":"anodal", "T4(2)_2":"cathodal"},
            "1038":{"T1":"sham", "T3":"anodal", "T4":"cathodal"},
            #"1042":{"T1":"sham", "T3":"anodal", "T4":"cathodal"},
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
        filepath = join(this_dir, f"NAP_{subj}_{k}.vhdr")
        raw = mne.io.read_raw_brainvision(filepath) # convert
        raw.save(join(proc_dir, f"NAP_{subj}_{v}-raw.fif"),
                 overwrite=overwrite)

