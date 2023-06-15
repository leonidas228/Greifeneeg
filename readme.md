# SFB Pipeline

## Run scripts in this sequence:

### convert.py
Grabs Brainvision raw data from a directory, converts to MNE-Python -raw.fif
and saves in another directory

### convert_sham.py
Grabs Brainvision raw sham data from a directory, converts to MNE-Python -raw.fif.
Specifically, these sham files have their sham stimulation periods already
marked by hand. Results are saved directly as *af*sham-raw.fif

### do_gssc.py
Grabs -raw.fif files, filters to 0.3-30Hz, runs GSSC and saves results either
as a CSV file or MNE-Python annotation file.

### dofilter.py
Grabs -raw.fif files, filters, and saves as f*-raw.fif files

### mark_stimulation.py
Grabs f*-raw.fif files, automatically identifies the stimulation type, and marks
stimulations with mne.Annotations. Saves as af*-raw.fif. Instead of having
T# (Tag) in the resulting filename, they will now have the identified condition
(e.g. Sham, eig30s, fix5m, etc).

### cutout_raw.py
Grabs af*-raw.fif files and cut away everything except the desired periods of
time before and/or after stimulation, save as caf*-raw.fif

### channel_organise.py
Grabs caf*-raw.fif files and makes sure channel and channel-types are properly
configured, save them as scaf*-raw.fif

### mark_badchans.py
Grabs scaf*-raw.fif files and runs an algorithmic bad channel detector, saves
as bscaf*-raw.fif

### doica.py
Grabs bscaf*-raw.fif, does ICA, and saves the solution under bscaf*-ica.fif

### icaclean.py
Grabs bscaf*-raw.fif and bscaf*-ica.fif, identifies and removes bad components,
saves as ibscaf*-raw.fif

### mark_osc.py
Grabs ibscaf*-raw.fif files and marks SO and DOs by the standard methodology.
Results are saved in two ways. The marked raw files are saved under
aibscaf*-raw.fif. Epoching is also done with the down state trough as the
0 point. These epochs are saved in one file, grand_CHANNEL-epo.fif, where
CHANNEL is the name of the ROI (typically "central"). This epoch file also
has an extensive metadata, which gives all sorts of information about each
epoch.

### epo_drop.py
Grabs the epo files output by mark_osc.py and drops noisy epochs. Outputs
with d_*-epo.fif
