import mne
from os import listdir
import re
import pickle
import numpy as np
from tensorpac import Pac, EventRelatedPac, PreferredPhase
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
conds = ["eig5m","fix5m","eig2m","fix2m","eig30s","fix30s","sham"]
conds = ["eig30s","fix30s"]
filelist = listdir(proc_dir)
chans = ["central"]
low_fq_range = list(np.linspace(0.5,1.25,50))
high_fq_range =  list(np.linspace(8,20,50))
fs = 500
n_jobs = 8
n_perm = 200

for filename in filelist:
    this_match = re.match("NAP_(\d{3})_(.*)-epo.fif",filename)
    if this_match:
        subj, cond = this_match.group(1), this_match.group(2)
        if cond not in conds:
            continue
        epo = mne.read_epochs(proc_dir+filename,preload=True)
        epo.resample(fs, n_jobs="cuda")
        for chan in chans:
            pick = mne.pick_channels(epo.ch_names, [chan])[0]
            data = epo.get_data()[:,pick,] * 1e+6

            # tensorpac
            p = Pac(f_pha=low_fq_range, f_amp=high_fq_range, dcomplex="wavelet")
            phases = p.filter(fs, data, ftype='phase', n_jobs=n_jobs)
            amplitudes = p.filter(fs, data, ftype='amplitude', n_jobs=n_jobs)
            p.idpac = (6, 1, 4)
            pac = p.fit(phases, amplitudes, n_jobs=n_jobs, n_perm=n_perm)
            pac_pickle = (p, pac)
            with open("{}{}_{}_{}_pac.pickle".format(proc_dir, subj, cond, chan), "wb") as f:
                pickle.dump(pac_pickle, f)
            p.idpac = (6, 0, 0)
            pac = p.fit(phases, amplitudes, n_jobs=n_jobs)
            pac_pickle = (p, pac)
            with open("{}{}_{}_{}_raw_pac.pickle".format(proc_dir, subj, cond, chan), "wb") as f:
                pickle.dump(pac_pickle, f)

            erp = EventRelatedPac(f_pha=[0.5,1.25], f_amp=high_fq_range, dcomplex="wavelet")
            phases = erp.filter(fs, data, ftype='phase', n_jobs=n_jobs)
            amplitudes = erp.filter(fs, data, ftype='amplitude', n_jobs=n_jobs)
            erpac = erp.fit(phases, amplitudes, method="gc", n_perm=n_perm,
                            n_jobs=n_jobs)
            erpac_pickle = (erp, erpac)
            with open("{}{}_{}_{}_erpac.pickle".format(proc_dir, subj, cond, chan), "wb") as f:
                pickle.dump(erpac_pickle, f)
            erpac = erp.fit(phases, amplitudes, method="gc", n_jobs=n_jobs)
            erpac_pickle = (erp, erpac)
            with open("{}{}_{}_{}_raw_erpac.pickle".format(proc_dir, subj, cond, chan), "wb") as f:
                pickle.dump(erpac_pickle, f)

            pp = PreferredPhase(f_pha=[0.5,1.25], f_amp=high_fq_range, dcomplex="wavelet")
            pp_res = pp.fit(phases, amplitudes)
            pp_pickle = (pp, pp_res)
            with open("{}{}_{}_{}_pp.pickle".format(proc_dir, subj, cond, chan), "wb") as f:
                pickle.dump(pp_pickle, f)
