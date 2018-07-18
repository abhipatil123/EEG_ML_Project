from braindecode.experiments.monitors import compute_preds_per_trial_for_set
from sklearn.metrics import roc_auc_score
import numpy as np
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