import mne
from os import listdir
import re

root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s"]
filelist = listdir(proc_dir)
#conds = ["eig5m"]
#subjs = ["031"]
excludes = ["031_eig30s", "045_fix5m"]

for filename in filelist:
    this_match = re.match("af_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if (cond not in conds) or ("{}_{}".format(subj,cond) in excludes):
            continue
        raw = mne.io.Raw(proc_dir+filename,preload=True)
        event_dict = {}
        for annot in raw.annotations:
            this_match = re.search("Pre_Stimulation_(\d)", annot["description"])
            if this_match:
                event_dict[annot["description"]] = int(this_match.group(1)) + 10
            this_match = re.search("Post_Stimulation_(\d)", annot["description"])
            if this_match:
                event_dict[annot["description"]] = int(this_match.group(1)) + 20
        events = mne.events_from_annotations(raw,event_dict)
        epo = mne.Epochs(raw,events[0],event_id=events[1],baseline=None,tmin=0,
                         tmax=60,reject_by_annotation=False)
        epo.save("{}af_NAP_{}_{}-epo.fif".format(proc_dir,subj,cond),
                 overwrite=True)
