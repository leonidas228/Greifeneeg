import mne
from os import listdir, getcwd
from os.path import isdir, join
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from numpy.polynomial import Polynomial as poly
from meegkit import detrend as dtr
from annot_stim import annot_stim

"""
Filter, resample, and organise the channels
"""


def robust_detrending(chan,deg = 1, it= 5, thresh=0.1, meegkit=False):

    if meegkit: 
        return dtr.detrend(x, order, n_iter)[0]
    else:
        weight = np.ones(len(chan))
        times = np.arange(len(chan))
        for p in range(it):    
            temp = poly.fit(times ,chan,deg, w=weight)
            val = temp.linspace(len(times))[1]
            dt = abs(val-chan)
            std = np.repeat(np.std(dt), len(chan))
            weight = np.where (dt/std > thresh, 0,weight)

        fitted = poly.fit(times, chan, deg,w=weight).linspace(len(times))[1]
        return chan-fitted




def annot_grad(raw, thresh=None, extend = 0.2, channel= None, start= 0, stop = None):
    data = raw.get_data(return_times=True, picks = channel, start= start, stop=stop)
    times = data[1]
    if channel:
        channel_list = channel
    else:
        channel_list = raw.ch_names
    annot_dict = dict(onset= list(), duration=list(), description = list(), orig_time = list(), ch_names = list())
    if thresh: 
 
        bad_jumps= mne.preprocessing.annotate_amplitude(raw, peak=thresh, min_duration=0.005,picks=channel)
        annot_bad_jumps=bad_jumps[0]
        annot_bad_jumps.onset-=0.2
        annot_bad_jumps.duration+=0.4  
        return annot_bad_jumps
    else:
        pend = np.zeros((len(channel_list),1))
        derive = np.diff(data[0], axis=1,prepend=pend)
        median = np.median(derive, axis=1)
        firstq, thirdq = np.percentile(derive, [25, 75], axis = 1)
        interquar = thirdq - firstq
        thresh = median + 20*interquar
    mask = np.zeros(np.shape(derive))
    mask = np.where(abs(derive)>thresh[:, None], 1, mask)
    segm = np.diff(mask, axis=1, append = pend)
    onsets = np.argwhere(segm ==1)
    offsets = np.argwhere(segm == -1)
    new_on = [[] for _ in range(len(channel_list))]

    new_off = [[] for _ in range(len(channel_list))]
    for i in range(len(offsets)):
        new_on[onsets[i][0]].append(onsets[i][1])
        new_off[offsets[i][0]].append(offsets[i][1])

    for i, chan in enumerate(channel_list):
        annot_dict['onset']+=list(times[new_on[i]]-extend)
        annot_dict['duration']+=list(times[new_off[i]]-times[new_on[i]]+2*extend)
        annot_dict['description']+=list(np.repeat('BAD_amplitude', len(new_on[i])))
        annot_dict['orig_time']+=list(np.repeat(raw.info['meas_date'], len(new_on[i])))
        annot_dict['ch_names']+=list(np.tile(channel_list[i], (len(new_on[i]),1)))
    for key, value in annot_dict.items():
        annot_dict[key] = np.asarray(value)
    annot= mne.Annotations(onset= annot_dict['onset'], duration= annot_dict['duration'], description= annot_dict['description'], orig_time = raw.info['meas_date'])

    return annot



def annot_abs(raw, thresh= 7.5e-4, extend = 0.2, channel= None, start = 0, stop = None):
    data = raw.get_data(return_times=True, picks = channel, start = start, stop= stop)
    times = data[1]
    if channel:
        channel_list = channel
    else:
        channel_list = raw.ch_names
    bad_ints = dict(onset= list(), duration=list(), description = list(), orig_time = list(), ch_names = list())
    

    mask = np.zeros(np.shape(data[0]))
    mask = np.where(abs(data[0])>thresh, 1, mask)
    segm = np.diff(mask, axis=1)
    onsets = np.argwhere(segm == 1)
    offsets = np.argwhere(segm == -1)
    new_on = [[] for _ in range(len(channel_list))]
    new_off = [[] for _ in range(len(channel_list))]
    for i in range(len(offsets)):
        new_on[onsets[i][0]].append(onsets[i][1])
        new_off[offsets[i][0]].append(offsets[i][1])
    for i, chan in enumerate(channel_list):
        bad_ints['onset']+=list(times[new_on[i]]-extend)
        bad_ints['duration']+=list(times[new_off[i]]-times[new_on[i]]+2*extend)
        bad_ints['description']+=list(np.repeat('BAD_amplitude_'+chan, len(new_on[i])))
        bad_ints['orig_time']+= list(np.repeat(raw.info['meas_date'], len(new_on[i])))
        bad_ints['ch_names']+=list(np.tile(channel_list[i], (len(new_on[i]),1)))
    for key, value in bad_ints.items():
        bad_ints[key] = np.asarray(value)
    annot= mne.Annotations(onset= bad_ints['onset'], duration= bad_ints['duration'], description= bad_ints['description'], ch_names = bad_ints['ch_names'], orig_time = raw.info['meas_date'])
    return annot

def artifact_annot(raw, channel = None, start = 0, stop = None):
    annotations= []
    #annotations.append(annot_grad(raw, channel= channel, start = start, stop = stop))
    #annotations.append(annot_abs(raw, channel = channel, start = start, stop = stop))
    #annotations.append(annot_abs(raw, thresh=15e-5, channel = channel+['Mov'], start = start, stop = stop))
    
    annotations.append(annot_stim(raw)[1])
    annot_all=annotations[0]
    for i in range(1,len(annotations)):
        annot_all = annot_all.__add__(annotations[i])
    raw.set_annotations(annot_all)

def data_by_annot(raw, description):
    annots = raw.annotations.copy()

    annot_idx= []
    for i in range(len(annots)):
        if annots[i]['description'] in description:
            annot_idx.append((annots[i]['onset'],annots[i]['onset']+annots[i]['duration'])) 
    print(annot_idx) 
    return annot_idx


#def prep_interv(raw, 


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
   
    artifact_annot(raw, channel=['Fz'], start = 0, stop = None)


    #data= dtr.detrend(raw.get_data()[0],1,n_iter = 5)[0]
    #raw._data=data

    #raw.apply_function(wrap, ['Fp1'], channel_wise=True, **kwargs)
    
    #artifact_annot(raw, channel = ['Fp1'],other= [annot_st]) 
        #raw.set_annotations(bad_mov)   
    raw.plot(block=True)# start = times[0], duration=times[1])

    break 
