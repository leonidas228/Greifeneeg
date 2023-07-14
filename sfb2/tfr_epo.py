import mne
from os import listdir
import re
from os.path import join
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
plt.ion()



def norm_overlay(x, min=10, max=20, centre=15, xmax=4e-05):
    x = x / xmax
    x = x * (max-min)/2 + centre
    return x


root_dir = "/home/jev/hdd/sfb2/"
proc_dir = join(root_dir, "proc")

chan = "central"
freqs = np.linspace(10, 20, 50)
n_cycles = 5
n_jobs = 24

epo = mne.read_epochs(join(proc_dir, f"grand_{chan}-epo.fif"))
epo = epo["OscType=='SO'"]
tfr = tfr_morlet(epo, freqs, n_cycles, return_itc=False, average=False, output="power",
                n_jobs=n_jobs)
tfr.crop(-2.25, 2.25)
tfr.apply_baseline((-2.25, -1), mode="zscore")

# get SO for overlays
epo.crop(-2.25, 2.25)
so_an = norm_overlay(epo["Cond=='anodal'"].average().data.squeeze())
so_ca = norm_overlay(epo["Cond=='cathodal'"].average().data.squeeze())
so_sh = norm_overlay(epo["Cond=='sham'"].average().data.squeeze())

line_color = "black"
mos_str = """
          AAXXDD
          AACCDD
          BBCCEE
          BBYYEE
          """
fig, axes = plt.subplot_mosaic(mos_str, figsize=(38.4, 21.6))
vmin, vmax = -3., 3
tfr_an = tfr["Cond=='anodal'"].average()
tfr_ca = tfr["Cond=='cathodal'"].average()
tfr_sh = tfr["Cond=='sham'"].average()

tfr_an.plot(vmin=0, vmax=vmax, axes=axes["A"], cmap="hot")
axes["A"].set_title("Anodal")
axes["A"].plot(tfr_an.times, so_an, color=line_color)

tfr_ca.plot(vmin=0, vmax=vmax, axes=axes["B"], cmap="hot")
axes["B"].set_title("Cathodal")
axes["B"].plot(tfr_ca.times, so_ca, color=line_color)

tfr_sh.plot(vmin=0, vmax=vmax, axes=axes["C"], cmap="hot")
axes["C"].set_title("Sham")
axes["C"].plot(tfr_sh.times, so_sh, color=line_color)

(tfr_an-tfr_sh).plot(vmin=vmin, vmax=vmax, axes=axes["D"])
axes["D"].set_title("Anodal - sham")
axes["D"].plot(tfr_an.times, so_an, color=line_color)

(tfr_ca-tfr_sh).plot(vmin=vmin, vmax=vmax, axes=axes["E"])
axes["E"].set_title("Cathodal - sham")
axes["E"].plot(tfr_ca.times, so_ca, color=line_color)

axes["X"].axis("off")
axes["Y"].axis("off")
plt.tight_layout()