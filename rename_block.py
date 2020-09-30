from os import listdir
import shutil
import re

root_dir = "/home/jev/hdd/sfb/"
proc_dir = root_dir+"proc/"
cond_codes = {"eigen5min":"eig5m","0.755min":"fix5m","eigen2min":"eig2m",
             "0.752min":"fix2m","eigen30s":"eig30s","0.7530s":"fix30s",
             "sham":"sham"}

with open("Randomisierung_naps.csv","rt") as f:
    naps = f.readlines()

sub_cond_codes = {}
this_subject = None
for nap in naps:
    line_list = nap.split("\t")
    line_list = [ll.replace(" ","") for ll in line_list]
    if line_list[1] == "": # blank line; skip
        continue
    if line_list[0] != "" and line_list[0] != this_subject: # new subject
        this_subject = line_list[0]
    if this_subject in sub_cond_codes:
        sub_cond_codes[this_subject].append(cond_codes[line_list[1]])
    else:
        sub_cond_codes[this_subject] = [cond_codes[line_list[1]]]

filelist = listdir(proc_dir)
for filename in filelist:
    this_match = re.match("f_(NAP_\d{3})_T(\d)-raw.fif",filename)
    if this_match:
        subj, tag = this_match.group(1), int(this_match.group(2))
        cond_name = sub_cond_codes[subj][tag-2]
        new_name = "{}f_{}_{}-raw.fif".format(proc_dir,subj,cond_name)
        print("{}{}\n{}\n\n".format(proc_dir,filename,new_name))
        shutil.move(proc_dir+filename,new_name)
