from os.path import isdir
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
font = {'weight' : 'bold',
        'size'   : 40}
matplotlib.rc('font', **font)
import warnings
warnings.filterwarnings("ignore")

def plot_werr(df, df_bs, x_order, hue_order, width=.3):
    fig, axis = plt.subplots(1,1, figsize=(21.6, 21.6))
    xtick_arr = np.arange(len(x_order))
    xticks = {"ticks":xtick_arr, "labels":x_order}
    for hue_idx, hue in enumerate(hue_order):
        df_h = df.query("Duration=='{}'".format(hue))
        df_bs_h = df_bs.query("Duration=='{}'".format(hue))
        vals = [df_h.query("Stimulation=='{}'".format(xo))["PAC"].values[0] for xo in x_order]
        vals_bs = [df_bs_h.query("Stimulation=='{}'".format(xo))["PAC"].values for xo in x_order]
        vals_bs = np.array(vals_bs)
        ci_low = np.quantile(vals_bs, 0.025, axis=1)
        ci_high = np.quantile(vals_bs, 0.975, axis=1)
        plt.bar(xtick_arr + width*(hue_idx-1), vals, width=width, label=hue,
                yerr=ci_high-ci_low)
    plt.xticks(**xticks)

    return fig, axis




if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/sfb/"
elif isdir("/home/jeff"):
    root_dir = "/home/jeff/hdd/jeff/sfb/"
proc_dir = root_dir+"proc/"

method= "wavelet"
baseline = "nobl"
baseline = "zscore"
time_win = (275,600)
#time_win = (50,600)
freq_win = (12, 15)
bootstrap = "load"

infile = "{}ModIdx_{}_{}_{}-{}Hz_{}-{}ms.pickle".format(proc_dir, method,
                                                        baseline, *freq_win,
                                                        *time_win)
print(infile)
df = pd.read_pickle(infile)

for osc in ["SO"]:
    for var in ["ND", "MVL", "PLV"]:
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

        # # model predictions
        # predict_df_dict = {"Stimulation":[],"Duration":[],"PAC":[]}
        # exog_names = mf.model.exog_names
        # for st in ["sham", "eig", "fix"]:
        #     for dur in ["30s", "2m", "5m"]:
        #         pac = mf.predict(exog={"StimType":[st], "Dur":[dur]}).values[0]
        #
        #         predict_df_dict["PAC"].append(pac)
        #         if st == "sham":
        #             predict_df_dict["Stimulation"].append("Sham")
        #         elif st == 'eig':
        #             predict_df_dict["Stimulation"].append("Eigen")
        #         elif st == 'fix':
        #             predict_df_dict["Stimulation"].append("Fixed")
        #         predict_df_dict["Duration"].append(dur)
        # predict_df = pd.DataFrame.from_dict(predict_df_dict)
        #
        # if bootstrap == "load":
        #     filename = "{}{}_{}_{}_{}-{}_{}-{}_predict_bs.pickle".format(proc_dir,
        #                                                                  var, osc,
        #                                                                  method,
        #                                                                  *freq_win,
        #                                                                  *time_win)
        #     predict_bs = pd.read_pickle(filename)
        # elif type(bootstrap) == int:
        #     predict_bs_dict = {"Stimulation":[],"Duration":[],"PAC":[]}
        #     subjs = list(np.unique(this_df["Subj"].values))
        #     for bs_idx in range(bootstrap):
        #         print("Bootstrap iteration {} of {}".format(bs_idx, bootstrap))
        #         this_df_bs = this_df.copy()
        #         for subj in subjs:
        #             subj_inds = this_df_bs["Subj"]==subj
        #             vals = this_df_bs[subj_inds][var].values
        #             inds = np.random.randint(len(vals), size=len(vals))
        #             this_df_bs.loc[subj_inds,var] = vals[inds]
        #
        #         mod_bs = smf.mixedlm(formula, data=this_df_bs, groups=this_df_bs["Sync"], re_formula=re_form, vc_formula=vc_form)
        #         mf_bs = mod_bs.fit(reml=False)
        #         exog_names = mf_bs.model.exog_names
        #         for st in ["sham", "eig", "fix"]:
        #             for dur in ["30s", "2m", "5m"]:
        #                 pac = mf_bs.predict(exog={"StimType":[st],
        #                                     "Dur":[dur]}).values[0]
        #                 predict_bs_dict["PAC"].append(pac)
        #                 if st == "sham":
        #                     predict_bs_dict["Stimulation"].append("Sham")
        #                 elif st == 'eig':
        #                     predict_bs_dict["Stimulation"].append("Eigen")
        #                 elif st == 'fix':
        #                     predict_bs_dict["Stimulation"].append("Fixed")
        #                 predict_bs_dict["Duration"].append(dur)
        #     predict_bs = pd.DataFrame.from_dict(predict_bs_dict)
        #     predict_bs.to_pickle("{}{}_{}_{}_{}-{}_{}-{}_predict_bs.pickle".format(proc_dir, var, osc, method, *freq_win, *time_win))
        #
        # fig, ax = plot_werr(predict_df, predict_bs, ["Sham", "Eigen", "Fixed"],
        #                     ["30s", "2m", "5m"])
        # plt.ylim((0.15, 0.23))
        # ax.text(2, 0.22, "*", fontsize=75)
        # #plt.legend()
        # plt.ylabel("normalised direct PAC", fontsize=44)
        # plt.suptitle("{} PAC LME model predictions, {}-{}Hz, {}-{}ms".format(osc, freq_win[0],
        #                                                        freq_win[1],
        #                                                        time_win[0],
        #                                                        time_win[1]))
        # plt.savefig("../images/{}_{}_{}_{}-{}_{}-{}_predict.png".format(var, osc, method, *freq_win, *time_win))
        # plt.savefig("../images/{}_{}_{}_{}-{}_{}-{}_predict.svg".format(var, osc, method, *freq_win, *time_win))
