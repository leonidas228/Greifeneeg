# SFB2 Pipeline

## Run scripts in this sequence:

### convert.py
Grabs Brainvision raw data from a directory, converts to MNE-Python -raw.fif
and saves in another directory with correct condition label

### prep.py
Grabs *-raw.fif files, filters, creates EOG/EMG channels, marks bad channels, 
and saves as p*-raw.fif files

### mark_stimulation.py
Grabs *p-raw.fif files, and marks stimulations on the basis of triggers. Saves as annotations.

### mark_sham_stimulation_algo.py
Grabs *p-raw.fif files, and marks sham stimulations on the basis of an algorithm. Saves as annotations.

### cutout_raw.py
Grabs *p-raw.fif and annotation files and cut away everything except the desired periods of
time before and/or after stimulation, save as cp*-raw.fif

### mark_osc.py
Grabs cp*-raw.fif files and marks SO and DOs by the standard methodology.
Results are saved in two ways. The marked raw files are saved under
aibscaf*-raw.fif. Epoching is also done with the down state trough as the
0 point. These are saved both in raw and epoched format, *-CHANNEL-epo/raw.fif, where
CHANNEL is the name of the ROI. The epoch files also
have an extensive metadata (found under Epochs.metadata), which gives all sorts of information about each
epoch.

### epo_cat.py
Combines the epo files from mark_osc into one, grand_CHANNEL-epo.fif file

### tfr_epo.py
Do a TFR analysis on the grand_epo SOs, compare anodal, cathodal, sham in a figure

# erpac.py
Do an ERPAC analysis on the grand epo SOs, compare anodal, cathodal, sham in separate figures

# erpac_sham.py
Use ERPAC to compare anodal and cathodal sham with each other.

# ttest_sham.py 
Subjects-average, paired-sample t-test of the sham conditions with other, with multiple comparisons correction

# ttest.py 
Subjects-average, paired-sample t-test of the anodal and cathodal stimulation against anodal sham, with multiple comparisons correction.

# slope_est.py
Calculate the slope of the PSD for sham and stim conditions.

