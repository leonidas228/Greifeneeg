import mne
from os import listdir
import re
import numpy as np
from mne.preprocessing import read_ica
import matplotlib.pyplot as plt
plt.ion()
from os.path import isdir

class Cycler():
    def __init__(self,filenames,proc_dir,picks,l_freq,h_freq):
        self.filenames = filenames
        self.proc_dir = proc_dir
        self.picks = picks
        self.l_freq = l_freq
        self.h_freq = h_freq
    def go(self):
        plt.close("all")
        epofile = self.filenames.pop()
        print("{}\n".format(epofile))
        self.epo = mne.read_epochs(self.proc_dir+epofile,preload=True)
        try:
            self.e_SO = self.epo["OscType=='SO'"].filter(l_freq=self.l_freq, h_freq=self.h_freq)
            self.e_deltO = self.epo["OscType=='deltO'"].filter(l_freq=self.l_freq, h_freq=self.h_freq)
        except:
            print("Incomplete set of oscillations")
            return
        #self.e_SO.plot(picks=self.picks,decim=1)
        self.e_SO.plot_image(picks=self.picks)
        #self.e_deltO.plot(picks=self.picks,decim=1)
        self.e_deltO.plot_image(picks=self.picks)
        self.evo_SO = self.e_SO.average()
        self.evo_SO.comment = "SO"
        self.evo_deltO = self.e_deltO.average()
        self.evo_deltO.comment = "deltO"
        mne.viz.plot_compare_evokeds([self.evo_SO,self.evo_deltO],picks=self.picks)
        print("\n{} Slow Oscillations\n{} Delta Oscillations\n".format(len(self.e_SO),len(self.e_deltO)))


if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
conds = ["fix30s", "eig30s"]
filelist = listdir(proc_dir)

filenames = []
for filename in filelist:
    this_match = re.match("d_NAP_(\d{3})_(.*)-epo.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        filenames.append(filename)

cyc = Cycler(filenames,proc_dir,"central",l_freq=None, h_freq=3)
