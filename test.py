
import glob
import os.path
subject_id = 1
base_path = "/home/ideatower/PycharmProjects/EEG_Project/physionet.org/pn6/chbmit/"
edf_file_names = sorted(glob.glob(os.path.join(base_path, "chb{:02d}/*.edf".format(subject_id))))
summary_file = os.path.join(base_path, "chb{:02d}/chb{:02d}-summary.txt".format(subject_id, subject_id))
summary_content = open(summary_file,'r').read()

import re
import mne
import numpy as np


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

all_X = []
all_y = []
for edf_file_name in edf_file_names:
    X, y = extract_data_and_labels(edf_file_name, summary_content)
    all_X.append(X)
    all_y.append(y)

dummy_X = all_X#[x[:,:2000] for x in all_X]

dummy_y = all_y#[y[:2000] for y in all_y]

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import split_into_two_sets
whole_set = SignalAndTarget(dummy_X, dummy_y) #Just an object which includes X and Y
train_set, test_set = split_into_two_sets(whole_set,0.5)
train_set, valid_set = split_into_two_sets(train_set, 0.7)

#Modeling of data


from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_time_length = 1200
n_classes = 2
in_chans = train_set.X[0].shape[0]
# final_conv_length determines the size of the receptive field of the ConvNet
#model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length,
#                        final_conv_length=12).create_network()
model = Deep4Net(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length,
                final_conv_length=2, stride_before_pool=True).create_network()
to_dense_prediction_model(model)

if cuda:
    model.cuda()

from torch import optim

optimizer = optim.Adam(model.parameters())



from braindecode.torch_ext.util import np_to_var
# determine output size
test_input = np_to_var(np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
if cuda:
    test_input = test_input.cuda()
out = model(test_input)
n_preds_per_input = out.cpu().data.numpy().shape[2]
print("{:d} predictions per input/trial".format(n_preds_per_input))

from braindecode.experiments.monitors import compute_preds_per_trial_for_set

from sklearn.metrics import roc_auc_score


class SeizureMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        # this will be timeseries of predictions
        # for each trial
        preds_per_trial = compute_preds_per_trial_for_set(all_preds, self.input_time_length,
                                                          dataset)
        seizure_preds = []
        all_preds = []
        all_y = []
        for i_trial in range(len(preds_per_trial)):
            this_y = dataset.y[i_trial]
            this_preds = preds_per_trial[i_trial]
            this_preds = np.exp(this_preds[1])
            n_missing_preds = len(this_y) - len(this_preds)
            this_preds = np.concatenate((np.zeros(n_missing_preds, dtype=this_preds.dtype),
                                         this_preds))
            all_preds.extend(this_preds)
            all_y.extend(this_y)
            if np.any(this_y == 1):
                seizure_preds.append(this_preds[this_y == 1])
        if len(seizure_preds) > 0:
            max_seiz_preds = np.array([np.max(p) for p in seizure_preds])
            sensitivity = np.mean(max_seiz_preds > 0.5)
        else:
            sensitivity = np.nan

        sensitivity_name = "{:s}_sensitivity".format(setname)
        if len(np.unique(all_y)) > 1:
            auc = roc_auc_score(all_y, all_preds)
        else:
            auc = np.nan
        auc_name = "{:s}_auc".format(setname)
        return {sensitivity_name: float(sensitivity),
                auc_name: float(auc)}

from braindecode.torch_ext.losses import log_categorical_crossentropy
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor, CroppedTrialMisclassMonitor, MisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs
import torch.nn.functional as F
import torch as th
from braindecode.torch_ext.modules import Expression
# Iterator is used to iterate over datasets both for training
# and evaluation
iterator = CropsFromTrialsIterator(batch_size=32,input_time_length=input_time_length,
                                  n_preds_per_input=n_preds_per_input)

# Loss function takes predictions as they come out of the network and the targets
# and returns a loss
loss_function = lambda preds, targets: log_categorical_crossentropy(preds, targets)
# Could be used to apply some constraint on the models, then should be object
# with apply method that accepts a module
model_constraint = None
# Monitors log the training progress
monitors = [LossMonitor(), MisclassMonitor(col_suffix='misclass'),
            SeizureMonitor(input_time_length),
            RuntimeMonitor(),]
# Stop criterion determines when the first stop happens
stop_criterion = MaxEpochs(5)
exp = Experiment(model, train_set, valid_set, test_set, iterator, loss_function, optimizer, model_constraint,
          monitors, stop_criterion, remember_best_column='valid_misclass',
          run_after_early_stop=True, batch_modifier=None, cuda=cuda)

# need to setup python logging before to be able to see anything
import logging
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
exp.run()