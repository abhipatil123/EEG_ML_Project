#Modeling of data
from splits import *
import numpy as np
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
