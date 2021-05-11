import mne
import matplotlib.pyplot as plt
from os.path import isdir
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 28}
matplotlib.rc('font', **font)

exclude = ["002", "003", "028"]

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

epo = mne.read_epochs("{}grand_central_finfo-epo.fif".format(proc_dir))
for excl in exclude:
    epo = epo["Subj!='{}'".format(excl)]

epo = epo.crop(tmin=-1.5, tmax=1.5)

e_SO = epo["OscType=='SO'"]
e_DO = epo["OscType=='deltO'"]

e_SO_dat = e_SO.get_data()[:,0] * 1e+6
e_DO_dat = e_DO.get_data()[:,0] * 1e+6

fig = plt.figure(figsize=(38.4, 21.6))
ax = plt.subplot(1,2,1)
ax.set_ylim((-200, 200))
ax.plot(epo.times, e_SO_dat.T, color="blue", alpha=0.005)
ax.plot(epo.times, e_SO_dat.mean(axis=0), color="black")
ax.set_title("Slow Oscillations, N={}".format(len(e_SO)))
ax.set_xlabel("Time (s)")
ax.set_ylabel("\u03BC"+"v")

ax = plt.subplot(1,2,2)
ax.set_ylim((-200, 200))
ax.plot(epo.times, e_DO_dat.T, color="red", alpha=0.005)
ax.plot(epo.times, e_DO_dat.mean(axis=0), color="black")
ax.set_title("Delta Oscillations, N={}".format(len(e_DO)))
ax.set_yticks([])
ax.set_xlabel("Time (s)")

plt.suptitle("Detected Slow Wave Sleep Oscillations")

plt.savefig("../images/oscs_erp.png")
plt.savefig("../images/oscs_erp.svg")
