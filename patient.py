
import glob
import os.path
import re
import mne
import numpy as np

#Get raw data of patient
subject_id = 1
base_path = "/home/ideatower/PycharmProjects/EEG_Project/"
edf_file_names = sorted(glob.glob(os.path.join(base_path, "chb{:02d}/*.edf".format(subject_id))))
summary_file = os.path.join(base_path, "chb{:02d}/chb{:02d}-summary.txt".format(subject_id, subject_id))
summary_content = open(summary_file,'r').read()


def extract_data_and_labels(edf_filename, summary_text):
    folder, basename = os.path.split(edf_filename)

    edf = mne.io.read_raw_edf(edf_filename, stim_channel=None)
    X = edf.get_data().astype(np.float32) * 1e6  # to mV
    y = np.zeros(X.shape[1], dtype=np.int64)
    i_text_start = summary_text.index(basename)

    if 'File Name' in summary_text[i_text_start:]:
        i_text_stop = summary_text.index('File Name', i_text_start)
    else:
        i_text_stop = len(summary_text)
    assert i_text_stop > i_text_start

    file_text = summary_text[i_text_start:i_text_stop]
    if 'Seizure Start' in file_text:
        start_sec = int(re.search(r"Seizure Start Time: ([0-9]*) seconds", file_text).group(1))
        end_sec = int(re.search(r"Seizure End Time: ([0-9]*) seconds", file_text).group(1))
        i_seizure_start = int(round(start_sec * edf.info['sfreq']))
        i_seizure_stop = int(round((end_sec + 1) * edf.info['sfreq']))
        y[i_seizure_start:i_seizure_stop] = 1
    assert X.shape[1] == len(y)
    return X, y

#Extract and append all data from edf files to X input and label as 
#seizure or non-seizure in y output
all_X = []
all_y = []
for edf_file_name in edf_file_names:
    X, y = extract_data_and_labels(edf_file_name, summary_content)
    all_X.append(X)
    all_y.append(y)

count_seizures = 0
count_nonseizures = 0
# print(all_X)
for i in range(len(all_y)):
    count_seizures += (all_y[i] == 1).sum()
    count_nonseizures += (all_y[i] == 0).sum()

print(count_nonseizures/count_seizures)
# print("No: of Seizure datapoints: " + count_seizures)
# print("No: of Non-Seizure datapoints: " + count_nonseizures)