import mne
from os import listdir, getcwd
from os.path import isdir, join
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from numpy.polynomial import Polynomial as poly

"""
Filter, resample, and organise the channels
"""




def robust_detrending(raw,thresh, channel=None):
    data = raw.get_data(return_times=True, picks= channel)
    times = np.asarray(data[1])
    dummy_times = np.arange(len(times))
    for chan in data[0]:
        print(len(chan))
        weight = np.ones((chan.shape[0]))
        print('here?')
        for p in range(5):    
            temp = poly.fit(dummy_times,chan,1, w=weight)
            val = temp.linspace(len(times))[1]
            dt = abs(val-chan)
            std = np.repeat(np.std(dt), len(chan))
            print(std)
            weight = np.where (dt/std > thresh, 0,weight)

        fitted = poly.fit(times, chan, 1,w=weight).linspace(len(times))[1]
        chan-=fitted


def amp_annot_grad(raw, thresh, extend = 0.2, channel= None):
    data = raw.get_data(return_times=True, picks = channel)
    times = data[1]
    if channel:
        channel_list = channel
    else:
        channel_list = raw.ch_names
    bad_ints = dict(onset= list(), duration=list(), description = list(), orig_time = list(), ch_names = list())
    

    for i, chan in enumerate(data[0]):
        bad = np.zeros(len(times))
        bad = np.where(abs(chan)>thresh, 1, bad)
        bad = np.concatenate(([0], bad, [0]))
        ann = np.argwhere((bad[:-1]+bad[1:])==1)
        print(ann[0])
        ann = ann[:,0]
        print(ann[0])
        bad_ints['onset']+=list(times[ann[::2]]-extend)
        bad_ints['duration']+=list(times[ann[1::2]]-times[ann[::2]]+2*extend)
        bad_ints['description']+=list(np.repeat('BAD_amplitude', len(ann[::2])))
        bad_ints['orig_time']+=list(np.repeat(raw.info['meas_date'], len(ann[::2])))
        bad_ints['ch_names']+=list(np.tile(channel_list[i], (len(ann[::2]),1)))
    for key, value in bad_ints.items():
        bad_ints[key] = np.asarray(value)
        
        derive = np.gradient(chan)
        peaks = scp.signal.find_peaks(derive)[0]
        print(len(chan))
        print('++++++++++++++++++')
        print(len(peaks))
        bad_int = dict(onset= list(), duration=list())
        for p in range(len(peaks)-1):
            max_int = np.max(chan[peaks[p]:peaks[p+1]])
            diff = abs(max_int-chan[peaks[p]])
            if diff > thresh:
                bad_int['onset'].append(times[peaks[p]]-extend)
                bad_int['duration'].append(times[peaks[p]]-times[peaks[p+1]]+2*extend)

        description = "BAD_amplitude"
        orig_time = raw.info['meas_date']
    annot= mne.Annotations(onset= bad_int['onset'], duration= bad_int['duration'], description= 'bad_int', orig_time = raw.info['meas_date'])
    return annot


def amp_annot_abs(raw, thresh, extend = 0.2, channel= None):
    data = raw.get_data(return_times=True, picks = channel)
    times = data[1]
    if channel:
        channel_list = channel
    else:
        channel_list = raw.ch_names
    bad_ints = dict(onset= list(), duration=list(), description = list(), orig_time = list(), ch_names = list())
    

    for i, chan in enumerate(data[0]):
        bad = np.zeros(len(times))
        bad = np.where(abs(chan)>thresh, 1, bad)
        bad = np.concatenate(([0], bad, [0]))
        ann = np.argwhere((bad[:-1]+bad[1:])==1)
        ann = ann[:,0]
        bad_ints['onset']+=list(times[ann[::2]]-extend)
        bad_ints['duration']+=list(times[ann[1::2]]-times[ann[::2]]+2*extend)
        bad_ints['description']+=list(np.repeat('BAD_amplitude', len(ann[::2])))
        bad_ints['orig_time']+= list(np.repeat(raw.info['meas_date'], len(ann[::2])))
        bad_ints['ch_names']+=list(np.tile(channel_list[i], (len(ann[::2]),1)))
    for key, value in bad_ints.items():
        bad_ints[key] = np.asarray(value)
    annot= mne.Annotations(onset= bad_ints['onset'], duration= bad_ints['duration'], description= bad_ints['description'], ch_names = bad_ints['ch_names'], orig_time = raw.info['meas_date'])
    return annot

def artifact_annot(raw):

    bad_jumps= mne.preprocessing.annotate_amplitude(raw, peak=15e-5, min_duration=0.005, picks = ['Fp1'])
    annot_bad_jumps=bad_jumps[0]
    annot_bad_jumps.onset-=0.2
    annot_bad_jumps.duration+=0.4  

    annot_bad_int = amp_annot_abs(raw, thresh=5e-4, channel = ['Fp1'])
    annot_bad_mov = amp_annot_abs(raw, thresh=15e-5, channel = ['Mov'])
    
    annot_all = annot_bad_int.__add__(annot_bad_jumps.__add__(annot_bad_mov))

    raw.set_annotations(annot_bad_jumps)






root_dir = getcwd()
proc_dir = join(root_dir, "data/proc")

l_freq = 0.1
h_freq = 100
n_jobs = "cuda" # change this to 1 or some higher integer if you don't have CUDA
sfreq = 200

overwrite = True
filelist = listdir(proc_dir)
sfreqs = {}
for filename in filelist:
    this_match = re.match("p_NAP_(\d{4})_(.*)-raw.fif", filename)
    if not this_match:
        continue
    (subj, cond) = this_match.groups()
    outfile = f"p_NAP_{subj}_{cond}-raw.fif"
    if outfile in filelist and not overwrite:
        print("Already exists. Skipping.")
        continue


    raw = mne.io.Raw(join(proc_dir, filename), preload=True)
    other = raw.copy()
    names = raw.ch_names
    names.remove('Fp1')
    print(other.ch_names)
    print(names)
    other.drop_channels(names)
    other.rename_channels({'Fp1':'other'})
    print(other.ch_names)
    raw.add_channels([other], force_update_info=True)
    robust_detrending(raw, 0.5, ['Fp1'])
    #artifact_annot(raw) 
    
        #raw.set_annotations(bad_mov)   
    #bad= merge_annotations(bad_jumps, bad_mov)
    #print(bad_mov) 
    #bad= bad_jumps.__add__(bad_mov)
    #raw.set_annotations(bad)
    raw.plot(block=True, order=[0,-1])

    break 
