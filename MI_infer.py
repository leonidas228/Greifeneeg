from os.path import isdir
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 28}
matplotlib.rc('font', **font)


if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

method="wavelet"
baseline = "nobl"
#baseline = "zscore"
time_win = (150,600)
#time_win = (50,600)
freq_win = (12, 15)

infile = "{}ModIdx_{}_{}_{}-{}Hz_{}-{}ms.pickle".format(proc_dir, method,
                                                        baseline, *freq_win,
                                                        *time_win)
print(infile)
df = pd.read_pickle(infile)

for osc in ["SO"]:
    for var in ["ND"]:
        fig, ax = plt.subplots(figsize=(38.4, 21.6))
        this_df = df.query("OscType=='{}'".format(osc))

        sns.barplot(data=this_df, y=var, x="StimType", hue="Dur",
                    order=["sham", "eig", "fix"], hue_order=["30s", "2m", "5m"],
                    ax=ax)
        plt.suptitle("{} {} ({} transformed)".format(osc, var, method))
        plt.savefig("../images/{}_{}_{}".format(var, osc, method))

        vc_form = {"Subj": "0 + C(Subj)"}
        re_form = "0 + Stim*Dur"
        #re_form = None
        formula = "{} ~ C(StimType, Treatment('sham'))*C(Dur, Treatment('30s'))".format(var)
        mod = smf.mixedlm(formula, data=this_df, groups=this_df["Sync"],
                          re_formula=re_form, vc_formula=vc_form)
        mf = mod.fit(reml=False)
        print(mf.summary())

        # model predictions
        predict_df_dict = {"Stimulation":[],"Duration":[],"PAC":[]}
        exog_names = mf.model.exog_names
        for st in ["sham", "eig", "fix"]:
            for dur in ["30s", "2m", "5m"]:
                pac = mf.predict(exog={"StimType":[st], "Dur":[dur]}).values[0]
                predict_df_dict["PAC"].append(pac)
                if st == "sham":
                    predict_df_dict["Stimulation"].append("Sham")
                elif st == 'eig':
                    predict_df_dict["Stimulation"].append("Eigen")
                elif st == 'fix':
                    predict_df_dict["Stimulation"].append("Fixed")

                predict_df_dict["Duration"].append(dur)

        predict_df = pd.DataFrame.from_dict(predict_df_dict)

        fig, ax = plt.subplots(figsize=(19.2, 19.2))
        sns.barplot(data=predict_df, y="PAC", x="Stimulation", hue="Duration",
                    order=["Sham", "Eigen", "Fixed"], hue_order=["30s", "2m", "5m"],
                    ax=ax)
        plt.legend(loc="upper left")
        plt.ylim((0.15, 0.25))
        plt.ylabel("normalised direct PAC", fontsize=36)
        plt.suptitle("{} PAC LME model predictions, {}-{}Hz, {}-{}ms".format(osc, freq_win[0],
                                                               freq_win[1],
                                                               time_win[0],
                                                               time_win[1]))
        plt.savefig("../images/{}_{}_{}_{}-{}_{}-{}_predict.png".format(var, osc, method, *freq_win, *time_win))
        plt.savefig("../images/{}_{}_{}_{}-{}_{}-{}_predict.svg".format(var, osc, method, *freq_win, *time_win))
