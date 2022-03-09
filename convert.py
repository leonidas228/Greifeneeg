import mne
from os import listdir
from os.path import isdir
import re

# different directories for home and office computers; not generally relevant
# for other users
if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"

raw_dir = root_dir+"raw/" # get raw files from here
proc_dir = root_dir+"proc/" # save the processed files here
filelist = listdir(raw_dir) # get list of all files in raw directory
proclist = listdir(proc_dir) # and in proc directory
overwrite = False# skip
do_subj = None

for filename in filelist: # cycle through all files in raw directory
    this_match = re.search("NAP_(\d{3})_T(\d)(b|c?).vhdr", filename)
    # do something if the file fits the raw file pattern
    if this_match:
        # pull subject and tag out of the filename and assign to variables
        subj, tag = this_match.group(1), this_match.group(2)
        if "NAP_{}_T{}-raw.fif".format(subj,tag) in proclist and not overwrite:
            print("Already exists. Skipping.")
            continue
        # if subj != do_subj:
        #     continue
        raw = mne.io.read_raw_brainvision(raw_dir+filename) # convert
        if "NAP_{}_T{}_2.vhdr".format(subj, tag) in filelist:
            print("Caught a _2 version.")
            raw_2 = mne.io.read_raw_brainvision(raw_dir+"NAP_{}_T{}_2.vhdr".format(subj, tag))
            raw.append([raw_2])
        raw.save("{}NAP_{}_T{}-raw.fif".format(proc_dir, subj, tag),
                 overwrite=overwrite) # save
