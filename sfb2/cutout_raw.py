import mne
from os import listdir
import re
from os.path import isdir, join

"""
cut away everything except the desired periods of time before and/or after stimulation
"""

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")

overwrite = False
filelist = listdir(proc_dir)
for filename in filelist:
    # cycle through filenames which match the pattern
    this_match = re.match("p_NAP_(\d{4})_(.*)-raw.fif", filename)
    if not this_match:
        continue
    (subj, cond) = this_match.groups()
    outfile = f"cp_NAP_{subj}_{cond}-raw.fif"
    if outfile in filelist and not overwrite:
        print("Already exists. Skipping.")
        continue
    # load up raw and annotations
    try:
        raw = mne.io.Raw(join(proc_dir, filename), preload=True)
        annots = mne.read_annotations(join(proc_dir, f"stim_NAP_{subj}_{cond}-annot.fif"))
    except:
        continue
    raw.set_annotations(annots)

    # special cases
    if subj == "1026" and cond == "anodal":
        raw.crop(tmax=4180)

    # run through all the annotations, cutting out the pre or post stimulation ones (ignore stimulation itself)
    raws = []
    for annot in annots:
        match = re.match("(.*)_Stimulation (\d.*)", annot["description"])
        if match:
            (stim_pos, stim_idx) = match.groups()
            if stim_pos == "BAD":
                continue
            # get the onset and duration times
            begin, duration = annot["onset"], annot["duration"]
            end = begin + duration
            # in case the annotation goes beyond the end of the recording (shouldn't happen anyway)
            if end > raw.times[-1]:
                end = raw.times[-1]
            try:
                raws.append(raw.copy().crop(begin, end))
            except:
                pass
    # if there were no pre/post stimulation periods
    if len(raws) == 0:
        continue
    # now merge into one file
    raw_cut = raws[0]
    raw_cut.append(raws[1:])
    raw_cut.save(join(proc_dir, outfile), overwrite=overwrite)

