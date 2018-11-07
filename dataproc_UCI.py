import numpy as np
import pandas as pd
from collections import Counter

DATASET_PATH = "./data/UCI HAR Dataset/"


INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]


def load_x(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        with open(signal_type_path, "r") as f:
            X_signals.append([np.array(serie, dtype=np.float32)
                             for serie in [row.replace('  ', ' ').strip().split(' ') for row in f]])
    return np.transpose(X_signals, (1, 2, 0))   # (?, 128, 9)


def load_y(y_path):
    with open(y_path, "r") as f:
        y = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in f]],
                     dtype=np.int32)
    y = y.reshape(-1, )
    return y - 1    # (?, 1)


train_x_signals_paths = [
    DATASET_PATH + "train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
test_x_signals_paths = [
    DATASET_PATH + "test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

train_y_path = DATASET_PATH + "train/y_train.txt"
test_y_path = DATASET_PATH + "test/y_test.txt"

train_x = load_x(train_x_signals_paths)
test_x = load_x(test_x_signals_paths)

train_y = load_y(train_y_path)
test_y = load_y(test_y_path)

train_y_one_hot = np.asarray(pd.get_dummies(train_y), dtype=np.int8)
test_y_one_hot = np.asarray(pd.get_dummies(test_y), dtype=np.int8)

print(train_x.shape, test_x.shape)
print(train_y.shape, Counter(train_y))
print(test_y.shape, Counter(test_y))
print(train_y_one_hot.shape)
print(test_y_one_hot.shape)

np.save("./data/processed/np_train_x.npy", train_x)
np.save("./data/processed/np_train_y.npy", train_y)
np.save("./data/processed/np_train_y_one_hot.npy", train_y_one_hot)
np.save("./data/processed/np_test_x.npy", test_x)
np.save("./data/processed/np_test_y.npy", test_y)
np.save("./data/processed/np_test_y_one_hot.npy", test_y_one_hot)

# (7352, 128, 9) (2947, 128, 9)
# (7352,) Counter({5: 1407, 4: 1374, 3: 1286, 0: 1226, 1: 1073, 2: 986})
# (2947,) Counter({5: 537, 4: 532, 0: 496, 3: 491, 1: 471, 2: 420})
# (7352, 6)
# (2947, 6)

