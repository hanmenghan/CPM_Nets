import scipy.io as sio
import numpy as np
import math
from numpy.random import shuffle
import tensorflow as tf

class DataSet(object):

    def __init__(self, data, view_number, labels):
        """
        Construct a DataSet.
        """
        self.data = dict()
        self._num_examples = data[0].shape[0]
        self._labels = labels
        for v_num in range(view_number):
            self.data[str(v_num)] = data[v_num]

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def read_data(str_name, ratio, Normal=1):
    """read data and spilt it train set and test set evenly
    :param str_name:path and dataname
    :param ratio:training set ratio
    :param Normal:do you want normalize
    :return:dataset and view number
    """
    data = sio.loadmat(str_name)
    view_number = data['X'].shape[1]
    X = np.split(data['X'], view_number, axis=1)
    X_train = []
    X_test = []
    labels_train = []
    labels_test = []
    if min(data['gt']) == 0:
        labels = data['gt'] + 1
    else:
        labels = data['gt']
    classes = max(labels)[0]
    all_length = 0
    for c_num in range(1, classes + 1):
        c_length = np.sum(labels == c_num)
        index = np.arange(c_length)
        shuffle(index)
        labels_train.extend(labels[all_length + index][0:math.floor(c_length * ratio)])
        labels_test.extend(labels[all_length + index][math.floor(c_length * ratio):])
        X_train_temp = []
        X_test_temp = []
        for v_num in range(view_number):
            X_train_temp.append(X[v_num][0][0].transpose()[all_length + index][0:math.floor(c_length * ratio)])
            X_test_temp.append(X[v_num][0][0].transpose()[all_length + index][math.floor(c_length * ratio):])
        if c_num == 1:
            X_train = X_train_temp;
            X_test = X_test_temp
        else:
            for v_num in range(view_number):
                X_train[v_num] = np.r_[X_train[v_num], X_train_temp[v_num]]
                X_test[v_num] = np.r_[X_test[v_num], X_test_temp[v_num]]
        all_length = all_length + c_length
    if (Normal == 1):
        for v_num in range(view_number):
            X_train[v_num] = Normalize(X_train[v_num])
            X_test[v_num] = Normalize(X_test[v_num])

    traindata = DataSet(X_train, view_number, np.array(labels_train))
    testdata = DataSet(X_test, view_number, np.array(labels_test))
    return traindata, testdata, view_number


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)
