import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pickle
from sklearn.model_selection import ParameterGrid
from os.path import isdir

if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"
fs = 500
nur_ba = True
vmin, vmax = -0.3, 0.3
pvmin, pvmax = -0.2, 0.2

if nur_ba:
    preposts = ["Pre", "Post"]
    col_n = 2 + 1
else:
    preposts = list(ParameterGrid({"stims":[0,1,2,3,4],"PrePost":["Pre", "Post"]}))
    col_n = 10

for osc in ["SO", "deltO"]:
    for freq in ["eig", "fix"]:
        for stimlen in ["30s"]:
            fig = plt.figure()
            plt.suptitle("{}{} {}".format(freq,stimlen,osc))
            p_reses = []
            for pps_idx, pps in enumerate(preposts):
                # pac
                if nur_ba:
                    infile = "{}{}{}_{}_{}_central_pac.pickle".format(proc_dir,freq,stimlen,
                                                                      osc, pps)
                    title = "{} stimulations".format(pps)
                else:
                    infile = "{}{}{}_{}_{}_{}_central_pac.pickle".format(proc_dir,freq,stimlen,
                                                                         pps["stims"],osc,pps["PrePost"])
                    title = "PAC {} Stimulation {}".format(pps["PrePost"],
                                                           pps["stims"])
                with open(infile, "rb") as f:
                    p, p_res = pickle.load(f)
                p_reses.append(p_res)
                plt.subplot(2,col_n,pps_idx+1)
                p.comodulogram(p_res.mean(-1), title=title, colorbar=True, vmin=vmin, vmax=vmax)

                # erpac
                # if nur_ba:
                #     infile = "{}{}{}_{}_{}_central_erpac.pickle".format(proc_dir,freq,stimlen,
                #                                                         osc, pps)
                # else:
                #     infile = "{}{}{}_{}_{}_{}_central_erpac.pickle".format(proc_dir,freq,stimlen,
                #                                                          pps["stims"],osc,pps["PrePost"])
                # with open(infile, "rb") as f:
                #     erp, erp_res = pickle.load(f)
                # plt.subplot(3,col_n,len(preposts)+pps_idx+1)
                # xvec = np.arange(erp_res.shape[-1])/fs
                # erp.pacplot(erp_res.squeeze(), xvec, erp.yvec, xlabel="Time (seconds)",
                #             ylabel="Amplitude frequency", cmap="Spectral_r")

                # preferred phase
                if nur_ba:
                    infile = "{}{}{}_{}_{}_central_pp.pickle".format(proc_dir,freq,stimlen,
                                                                     osc, pps)
                else:
                    infile = "{}{}{}_{}_{}_{}_central_pp.pickle".format(proc_dir,freq,stimlen,
                                                                        pps["stims"],osc,pps["PrePost"])
                with open(infile, "rb") as f:
                    pp, pp_res = pickle.load(f)
                subtup = (2,col_n,col_n+pps_idx+1)
                ampbin = np.squeeze(pp_res[0].mean(-1)).T
                pp.polar(ampbin, pp_res[2].T, pp.yvec, subplot=subtup)

            plt.subplot(2,col_n,pps_idx+2)
            p.comodulogram(p_reses[1].mean(-1) - p_reses[0].mean(-1),
                           title="Post - Pre Stimulations", vmin=-0.3, vmax=0.3)
