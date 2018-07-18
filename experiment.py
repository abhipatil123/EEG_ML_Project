
from braindecode.torch_ext.losses import log_categorical_crossentropy
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor, CroppedTrialMisclassMonitor, MisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs
import torch.nn.functional as F
import torch as th
from braindecode.torch_ext.modules import Expression
from seizure_monitor import *
from model import *
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