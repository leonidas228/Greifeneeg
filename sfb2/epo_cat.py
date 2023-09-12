import mne
from os import listdir
import re
from os.path import isdir, join
import pandas as pd

"""
concatenate all epochs into one file
"""

root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")

overwrite = True
filelist = listdir(proc_dir)
epos = []
for filename in filelist:
    this_match = re.match("osc_NAP_(\d{4})_(.*)_(.*)_(.*)_(.*)-epo.fif", filename)
    if not this_match:
        continue
    #(subj, cond, chan, osc_type) = this_match.groups()
    epo = mne.read_epochs(join(proc_dir, filename))
    epos.append(epo)

grand_epo = mne.concatenate_epochs(epos)
grand_epo.save(join(proc_dir, "grand-epo.fif"), overwrite=True)

