import mne
from os import listdir
import re
import pickle
import numpy as np
from tensorpac import Pac, EventRelatedPac, PreferredPhase
from os.path import isdir
from sklearn.model_selection import ParameterGrid

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)
chans = ["central"]
low_fq_range = list(np.linspace(0.5,1.25,50))
high_fq_range =  list(np.linspace(8,20,50))
fs = 500
n_jobs = 8
n_perm = 200

param_grid = {"Index":[0,1,2,3,4], "Cond":["eig30s", "fix30s"],
              "PrePost":["Pre", "Post"], "OscType":["SO", "deltO"]}
grob_param_grid =
all_params = list(ParameterGrid(param_grid))

epo = mne.read_epochs("{}grand-epo.fif".format(proc_dir))
for params in all_params:
    match_string = ""
    for k,v in params.items():
        match_string += "{} == '{}' and ".format(k,v)
    match_string = match_string[:-5] # don't want the last ' and '
    e = epo[match_string]
    for chan in chans:
        pick = mne.pick_channels(e.ch_names, [chan])[0]
        data = e.get_data()[:,pick,] * 1e+6

        # tensorpac
        p = Pac(f_pha=low_fq_range, f_amp=high_fq_range, dcomplex="wavelet")
        phases = p.filter(fs, data, ftype='phase', n_jobs=n_jobs)
        amplitudes = p.filter(fs, data, ftype='amplitude', n_jobs=n_jobs)
        p.idpac = (6, 3, 4)
        pac = p.fit(phases, amplitudes, n_jobs=n_jobs, n_perm=n_perm)
        pac_pickle = (p, pac)
        with open("{}{}_{}_{}_{}_{}_pac.pickle".format(proc_dir, *params.values(), chan), "wb") as f:
            pickle.dump(pac_pickle, f)
        p.idpac = (6, 0, 0)
        pac = p.fit(phases, amplitudes, n_jobs=n_jobs)
        pac_pickle = (p, pac)
        with open("{}{}_{}_{}_{}_{}_raw_pac.pickle".format(proc_dir, *params.values(), chan), "wb") as f:
            pickle.dump(pac_pickle, f)

        erp = EventRelatedPac(f_pha=[0.5,1.25], f_amp=high_fq_range, dcomplex="wavelet")
        phases = erp.filter(fs, data, ftype='phase', n_jobs=n_jobs)
        amplitudes = erp.filter(fs, data, ftype='amplitude', n_jobs=n_jobs)
        erpac = erp.fit(phases, amplitudes, method="gc", n_perm=n_perm,
                        n_jobs=n_jobs)
        erpac_pickle = (erp, erpac)
        with open("{}{}_{}_{}_{}_{}_erpac.pickle".format(proc_dir, *params.values(), chan), "wb") as f:
            pickle.dump(erpac_pickle, f)
        erpac = erp.fit(phases, amplitudes, method="gc", n_jobs=n_jobs)
        erpac_pickle = (erp, erpac)
        with open("{}{}_{}_{}_{}_{}_raw_erpac.pickle".format(proc_dir, *params.values(), chan), "wb") as f:
            pickle.dump(erpac_pickle, f)

        pp = PreferredPhase(f_pha=[0.5,1.25], f_amp=high_fq_range, dcomplex="wavelet")
        pp_res = pp.fit(phases, amplitudes)
        pp_pickle = (pp, pp_res)
        with open("{}{}_{}_{}_{}_{}_pp.pickle".format(proc_dir, *params.values(), chan), "wb") as f:
            pickle.dump(pp_pickle, f)
