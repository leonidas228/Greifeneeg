# SFB Pipeline

## Run scripts in this sequence:

### convert.py
Grabs Brainvision raw data from a directory, converts to MNE-Python -raw.fif
and saves in another directory

### dofilter.py
Grabs -raw.fif files, filters, and saves as f*-raw.fif files

### mark_stimulation.py
Grabs f*-raw.fif files, automatically identifies the stimulation type, and marks
stimulations with mne.Annotations. Saves as af*-raw.fif. Instead of having
T# (Tag) in the resulting filename, they will now have the identified condition
(e.g. Sham, eig30s, fix5m, etc). Note: Sham files will be saved as 
f*sham-raw.fif, not as af*sham-raw.fif. As they will be later marked with their
own script (sham_stim_mark.py).
