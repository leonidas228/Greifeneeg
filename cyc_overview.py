import mne
from os import listdir
import re
import numpy as np
from mne.preprocessing import read_ica
import matplotlib.pyplot as plt
plt.ion()

class Cycler():
    def __init__(self,input_dict,proc_dir):
        self.input_dict = input_dict
        self.proc_dir = proc_dir
        self.subj_order = list(input_dict.keys())
        self.subj_order.sort(reverse=True)
        self.cond_order = []
        self.this_subj = None
        self.this_cond = None
    def go(self):
        plt.close("all")
        try:
            self.this_cond = self.cond_order.pop()
        except:
            try:
                self.this_subj = self.subj_order.pop()
            except:
                raise ValueError("Already cycled through all files.")
            self.cond_order = list(input_dict[self.this_subj].keys())
            self.cond_order.sort(reverse=True)
            self.this_cond = self.cond_order.pop()

        rawfile = input_dict[self.this_subj][self.this_cond]["rawfile"]
        print("{}\n".format(rawfile))
        self.raw = mne.io.Raw(self.proc_dir+rawfile)
        self.raw.plot()
        if show_ica:
            icafile = input_dict[self.this_subj][self.this_cond]["icafile"]
            print("{}\n".format(icafile))
            self.ica = read_ica(self.proc_dir+icafile)
            self.ica.plot_components(picks=np.arange(16))
            self.ica.plot_properties(self.raw,picks=np.arange(5))
            self.ica.plot_sources(self.raw)
        print("\n")

root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
filelist = listdir(proc_dir)
excludes = ["031_eig30s", "045_fix5m"]
show_ica = False

input_dict = {}
for filename in filelist:
    this_match = re.search("ibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds or "{}_{}".format(subj,cond) in excludes:
            continue
        rawfile = "{}bscaf_NAP_{}_{}-raw.fif".format(proc_dir,subj,cond)
        icafile = "bscaf_NAP_{}_{}-ica.fif".format(subj,cond)
        if (rawfile not in filelist or icafile not in filelist) and show_ica:
            raise ValueError("Not a full pair for subject {}".format(subj))
        if subj not in input_dict:
            input_dict[subj] = {}
        input_dict[subj][cond] = {"rawfile":rawfile, "icafile":icafile}

cyc = Cycler(input_dict,proc_dir)
