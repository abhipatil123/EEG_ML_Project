
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import split_into_two_sets
from patient import all_X, all_y

#Split the X and Y into training and testing sets
dummy_X = all_X
dummy_y = all_y

whole_set = SignalAndTarget(dummy_X, dummy_y) #Just an object which includes X and Y
train_set, test_set = split_into_two_sets(whole_set,0.5)
train_set, valid_set = split_into_two_sets(train_set, 0.7)

# print(train_set)