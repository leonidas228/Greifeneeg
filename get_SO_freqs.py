import mne
from mne.time_frequency import psd_multitaper, psd_welch
import numpy as np
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/sfb/"
proc_dir = root_dir+"proc/"

epo = mne.read_epochs(proc_dir+"grand_central-epo.fif", preload=True)
epo = epo["OscType=='SO'"]
epo.filter(l_freq=0.16, h_freq=1.25, n_jobs="cuda")
#epo = epo.apply_baseline((None,None))
df = epo.metadata.copy()
subjs = list(df["Subj"].unique())
zero_idx = epo.time_as_index(0)

signals = epo.get_data()[:,0,].squeeze()

# need to add infinitesimals to zeros to prevent weird x-crossing bugs
for null_idx in list(np.where(signals==0)[0]):
    if null_idx:
        signals[null_idx] = 1e-16*np.sign(signals[null_idx-1])
    else:
        signals[null_idx] = 1e-16*np.sign(signals[null_idx+1])

osc_lens, osc_freqs = [], []
for epo_idx in range(len(signals)):
    signal = signals[epo_idx,]
    zero_x_inds = (np.where((signal[:-1] * signal[1:]) < 0)[0]) + 1
    if signal[0] < 0:
        neg_crosses = zero_x_inds[1::2]
    else:
        neg_crosses = zero_x_inds[::2]
    # get neg crossings around 0
    try:
        neg_cross_0 = neg_crosses[neg_crosses<zero_idx].max()
        neg_cross_1 = neg_crosses[neg_crosses>zero_idx].min()
        osc_lens.append(epo.times[neg_cross_1] - epo.times[neg_cross_0])
        osc_freqs.append(1/osc_lens[-1])
    except:
        osc_lens.append(np.nan)
        osc_freqs.append(np.nan)

breakpoint()

cond_specs = {"sham":{"formula":"Cond=='sham30s'"}}
for dur in ["30s", "2m", "5m"]:
    cond_specs["eig{}_pre".format(dur)] = {"formula":"Cond=='eig{}' and PrePost=='Pre'".format(dur)}
    for ind in range(5):
        cond_specs["eig{}_{}".format(dur,ind)] = {"formula":"Cond=='eig{}' and PrePost=='Post'".format(dur)}

subj_fmaxes = {}
for subj in subjs:
    subj_fmaxes[subj] = {}
    subj_epo = epo["Subj=='{}'".format(subj)]

    stim_raw = mne.io.Raw("{}bad_caf_NAP_{}_eig30s-raw.fif".format(proc_dir, subj))
    psds, freqs = psd_welch(stim_raw, fmin=0.5, fmax=1, n_jobs=8)
    psd = psds.mean(axis=0)
    subj_fmaxes[subj]["stim_freq"] = freqs[np.argmax(psds)]

    for k,v in cond_specs.items():
        this_epo = subj_epo[v["formula"]]
        if len(this_epo):
            psds, freqs = psd_welch(this_epo, fmin=0.5, fmax=1, n_jobs=8)
            psd = psds.mean(axis=0)
            subj_fmaxes[subj][k] = freqs[np.argmax(psd)]
        else:
            subj_fmaxes[subj][k] = np.nan
