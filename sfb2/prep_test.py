import mne
from os import listdir, getcwd
from os.path import isdir, join
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from numpy.polynomial import Polynomial as poly
from meegkit import detrend as dtr
from annot_stim import annot_stim

"""
Filter, resample, and organise the channels
"""


def robust_detrending(chan,deg = 1, n_iter= 5, thresh=0.1, meegkit=False):

    if meegkit: 
        return dtr.detrend(x, order, n_iter)[0]
    else:
        weight = np.ones(len(chan))
        times = np.arange(len(chan))
        for p in range(n_iter):    
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
    pend = np.zeros((len(channel_list),1))
    derive = np.diff(data[0], axis=1,prepend=pend, append= pend)[:,0:-1]
    if thresh==None: 
        median = np.median(derive, axis=1)
        firstq, thirdq = np.percentile(derive, [25, 75], axis = 1)
        interquar = thirdq - firstq
        thresh = median + 20*interquar
    mask = np.zeros(np.shape(derive))
    mask = np.where(abs(derive)>thresh[:, None], 1, mask)
    segm = np.diff(mask, axis=1, prepend = pend, append = pend)[:,0:-1]
    onsets = np.argwhere(segm ==1)
    offsets = np.argwhere(segm == -1)
    if len(onsets)!=len(offsets):
        print(np.shape(segm))
    print(len(onsets),len(offsets))
    new_on = [[] for _ in range(len(channel_list))]

    new_off = [[] for _ in range(len(channel_list))]
    for i in range(len(onsets)):
        new_on[onsets[i][0]].append(onsets[i][1])
    for i in range(len(offsets)):
        new_off[offsets[i][0]].append(offsets[i][1])

    for i, chan in enumerate(channel_list):
        if len(new_on[i]) > len(new_off[i]):
            new_off[i].append(new_on[i][-1])
        annot_dict['onset']+=list(times[new_on[i]]-extend)
        annot_dict['duration']+=list(times[new_off[i]]-times[new_on[i]]+2*extend)
        annot_dict['description']+=list(np.repeat('BAD_step_'+chan, len(new_on[i])))
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
    

    pend = np.zeros((len(channel_list),1))
    mask = np.zeros(np.shape(data[0]))
    mask = np.where(abs(data[0])>thresh, 1, mask)
    segm = np.diff(mask, axis=1, prepend= pend, append=pend)[:,0:-1]
    onsets = np.argwhere(segm == 1)
    offsets = np.argwhere(segm == -1)
    new_on = [[] for _ in range(len(channel_list))]
    new_off = [[] for _ in range(len(channel_list))]
    for i in range(len(onsets)):
        new_on[onsets[i][0]].append(onsets[i][1])
    for i in range(len(offsets)):
        new_off[offsets[i][0]].append(offsets[i][1])
    for i, chan in enumerate(channel_list):
        if len(new_on[i]) > len(new_off[i]):
            new_off[i].append(new_on[i][-1])
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
    old_annotations= raw.annotations.copy()
    new_annotations= []
    new_annotations.append(annot_grad(raw, channel= channel, start = start, stop = stop))
    new_annotations.append(annot_abs(raw, channel = channel, start = start, stop = stop))
    new_annotations.append(annot_abs(raw, thresh=15e-5, channel = ['Mov'], start = start, stop = stop))
    
    for i in range(len(new_annotations)):
        old_annotations = old_annotations.__add__(new_annotations[i])
    raw.set_annotations(old_annotations)

def data_by_annot(raw, description, extend=True, end_interval = True):
    times = raw.get_data(return_times = True)[1]
    annots = raw.annotations.copy()
    if extend:
        description = [desc+'.*' for desc in description]
    annot_idx= []
    for i in range(len(annots)):
        for j in range(len(description)):
            if re.match(description[j], annots[i]['description']):
                annot_idx.append((list(times).index(annots[i]['onset']),list(times).index(annots[i]['onset']+annots[i]['duration']), annots[i]['description']))
    if end_interval:
        annot_idx.append((annot_idx[-1][1]+1, len(times)-1, 'End'))
    return annot_idx


def apply_detrend(data, intervals, deg = 5, n_iter= 5, thresh=0.1, delta_t = None, meegkit=False):
    for i in range(len(intervals)):
        if delta_t:
            deg=math.ceil((intervals[i][1]-intervals[i][0])*delta_t/10)
            print(deg)
        data[intervals[i][0]:intervals[i][1]] = robust_detrending(data[intervals[i][0]:intervals[i][1]], deg=deg,n_iter=n_iter,thresh=thresh, meegkit=False)
    return data


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
    this_match = re.match("NAP_(\d{4})_(.*)-raw.fif", filename)
    if not this_match:
        continue
    (subj, cond) = this_match.groups()
    outfile = f"p_NAP_{subj}_{cond}-raw.fif"
    if outfile in filelist and not overwrite:
        print("Already exists. Skipping.")
        continue


    raw = mne.io.Raw(join(proc_dir, filename), preload=True)


    #raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
    raw.notch_filter(np.arange(50, h_freq+50, 50), n_jobs=n_jobs)
    
    raw.resample(sfreq, n_jobs=n_jobs)


     # create EOG/EMG chanenls
    if "HEOG" not in raw.ch_names:
        raw = mne.set_bipolar_reference(raw, "Li", "Re", ch_name="HEOG")
        raw.set_channel_types({"HEOG":"eog"})
    if "Mov" not in raw.ch_names:
        raw = mne.set_bipolar_reference(raw, "MovLi", "MovRe", ch_name="Mov")
        raw.set_channel_types({"Mov":"emg"})
    if "VEOG" not in raw.ch_names and "Vo" in raw.ch_names and "Vu" in raw.ch_names:
        raw = mne.set_bipolar_reference(raw, "Vo", "Vu", ch_name="VEOG")
        raw.set_channel_types({"VEOG":"eog"})

    raw.set_annotations(annot_stim(raw)[1]) 
    idx = data_by_annot(raw, ['Pre_Stimulation','Post_Stimulation'], end_interval=True)
    times = raw.get_data(return_times=True)[1]
    delta_t = times[1]-times[0]
    print(times)
    
    raw = raw.apply_function(apply_detrend, intervals = idx, meegkit = True, deg = 4,delta_t = delta_t)
    for i in range(len(idx)):
        artifact_annot(raw, start = idx[i][0], stop = idx[i][1])


    for i in range(len(idx)):
        file_name = 'p_NAP_'+str(subj)+'_'+str(cond)+'_'+ idx[i][2]+'_raw.fif'
        raw.save(join(proc_dir, file_name), overwrite=overwrite, tmin = times[idx[i][0]], tmax = times[idx[i][1]])
    #raw.save(join(proc_dir, outfile), overwrite=overwrite)


    raw.plot(block=True)# start = times[0], duration=times[1])

    break 
