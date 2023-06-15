import mne
from os import listdir
from os.path import join
import re
import numpy as np
from gssc.infer import EEGInfer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt
plt.ion()

root_dir = "/home/jev/hdd/sfb/"
hand_dir = "/home/jev/sfb/ALLES ZUSAMMEN/"

proc_dir = root_dir+"proc/"
filelist = listdir(proc_dir)
proclist = listdir(proc_dir) # and in proc directory
overwrite = False # skip

l_freq = 0.3
h_freq = 30

calc = False

if calc:
    eeginfer = EEGInfer()
    all_gssc, all_hand = [], []
    for filename in filelist:
        this_match = re.search("NAP_(\d{3})_T(\d)-raw.fif",filename)
        if this_match:
            subj, tag = this_match.group(1), this_match.group(2)
            raw = mne.io.Raw(proc_dir+filename, preload=True)
            raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs="cuda")



            # hand staging
            this_dir = join(hand_dir, f"!NAP_{subj}")
            # get the right filename
            for filename in listdir(this_dir):
                this_match = re.search(f"NAP_{subj}_T{tag}(b|c?).txt", filename)
                if this_match:
                    break
            inpath = join(this_dir, filename)

            with open(inpath, "rt") as f:
                stages = f.readlines()
            hand_stages = np.array([int(x[0]) for x in stages])

            # # special cases
            # if subj == "021" and tag == "2":
            #     raw.crop(tmin=0,tmax=5340)
            #     hand_stages = hand_stages[:5340//30]

            # gssc staging
            raw.pick_channels(["F3", "F4", "C3", "C4", "Li", "Re"])
            type_dict = {c:"eog" for c in ["Li", "Re"]}
            raw.set_channel_types(type_dict)
            gssc_stages, times = eeginfer.mne_infer(raw)

            if len(gssc_stages) != len(hand_stages):
                hand_stages = hand_stages[:len(gssc_stages)]

            all_gssc.extend(gssc_stages)
            all_hand.extend(hand_stages)


    with open(join(root_dir, "sleep_staging.pickle"), "wb") as f:
        pickle.dump({"gssc":all_gssc, "hand":all_hand}, f)

with open(join(root_dir, "sleep_staging.pickle"), "rb") as f:
    results = pickle.load(f)

# fix inconsistencies
hand, gssc = np.array(results["hand"]), np.array(results["gssc"])
hand[hand==4] = 3
hand[hand==5] = 4
gssc = gssc[hand!=8]
hand = hand[hand!=8]


conf_mat = confusion_matrix(gssc, hand, normalize="pred")
disp = ConfusionMatrixDisplay(conf_mat)
ax = plt.subplot(111)
disp.plot(colorbar=False, ax=ax)
ax.set_ylabel("GSSC")
ax.set_xlabel("Liliia")
plt.show()
