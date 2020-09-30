import mne
from os import listdir
import re
import numpy as np
from mne.preprocessing import read_ica
import matplotlib.pyplot as plt
plt.ion()
from tensorpac import Pac, EventRelatedPac, PreferredPhase

class Cycler():
    def __init__(self,input_dict,proc_dir):
        self.input_dict = input_dict
        self.proc_dir = proc_dir
        self.subj_order = list(input_dict.keys())
        self.subj_order.sort(reverse=True)
        self.subj_order.sort()
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
        #self.raw.plot()
        epofile = input_dict[self.this_subj][self.this_cond]["epofile"]
        print("{}\n".format(epofile))
        self.epo = mne.read_epochs(self.proc_dir+epofile)
        #self.epo.plot_topo_image()
        print("\n")
        for chan in chans:
            pick = mne.pick_channels(self.epo.ch_names, [chan])[0]
            events = self.epo.events
            events[:,2] = np.ones_like(events[:,2])
            events[:,0] -= self.raw.first_samp
            self.epo.resample(fs, n_jobs=2)
            data = self.epo.get_data()[:,pick,] * 1e+6
            self.data = data

            # # tensorpac
            # p = Pac(f_pha=low_fq_range, f_amp=high_fq_range, dcomplex="wavelet")
            # phases = p.filter(fs, data, ftype='phase', n_jobs=2)
            # amplitudes = p.filter(fs, data, ftype='amplitude', n_jobs=2)
            # p.idpac = (6, 0, 4)
            # pac = p.fit(phases, amplitudes, n_perm=200)
            # pac_avg = pac.mean(-1)
            # pvalues = p.infer_pvalues(p=0.05, mcp="maxstat")
            # pac_avg_ns = pac_avg.copy()
            # pac_avg_ns[pvalues<0.05] = np.nan
            # p.comodulogram(pac_avg_ns, cmap='gray')
            # pac_avg_s = pac_avg.copy()
            # pac_avg_s[pvalues>0.05] = np.nan
            # p.comodulogram(pac_avg_s, cmap='Spectral_r')

            erp = EventRelatedPac(f_pha=(.3,1.25), f_amp=np.linspace(8,20,50), dcomplex="wavelet")
            phases = erp.filter(fs, data, ftype='phase', n_jobs=2)
            amplitudes = erp.filter(fs, data, ftype='amplitude', n_jobs=2)
            erpac = erp.fit(phases, amplitudes, method="gc", n_perm=200)
            p_vals = erp.infer_pvalues()
            erpac_n = erpac.copy()
            erpac_n[p_vals==0] = np.nan
            erpac_s = erpac.copy()
            erpac_s[p_vals==1] = np.nan
            plt.figure()
            erp.pacplot(erpac_n.squeeze(), self.epo.times, erp.yvec, cmap="gray")
            erp.pacplot(erpac_s.squeeze(), self.epo.times, erp.yvec)
            self.results = [p_vals, erpac]

            pp = PreferredPhase(f_pha=(.3,1.25), f_amp=np.linspace(8,20,50), dcomplex="wavelet")
            b_amp, pps, polar_vec = pp.fit(phases, amplitudes)
            ampbin = np.squeeze(b_amp).mean(-1).T
            plt.figure()
            pp.polar(ampbin, polar_vec, pp.yvec)

root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s"]
filelist = listdir(proc_dir)
excludes = []
chans = ["central"]
low_fq_range = list(np.linspace(0.4,4,50))
high_fq_range =  list(np.linspace(10,20,50))
fs = 100

input_dict = {}
for filename in filelist:
    this_match = re.search("aibscaf_NAP_(\d{3})_(.*)-raw.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds or "{}_{}".format(subj,cond) in excludes:
            continue
        rawfile = "aibscaf_NAP_{}_{}-raw.fif".format(subj,cond)
        epofile = "d_NAP_{}_{}-epo.fif".format(subj,cond)
        if (rawfile not in filelist or epofile not in filelist):
            raise ValueError("Not a full pair for subject {}".format(subj))
        if subj not in input_dict:
            input_dict[subj] = {}
        input_dict[subj][cond] = {"rawfile":rawfile, "epofile":epofile}

cyc = Cycler(input_dict,proc_dir)
