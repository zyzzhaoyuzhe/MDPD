import numpy as np
import logging
from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(10)

class Crowd_Sourcing_Readers():
    def __init__(self, crowd_file, label_file):
        self.is_missing_value = False
        crowd_cache = self.read_data(crowd_file)
        truth_cache = self.read_label(label_file)
        crowd_item = set(row[0] for row in crowd_cache)
        crowd_worker = set(row[1] for row in crowd_cache)
        crowd_response = set(row[2] for row in crowd_cache)
        truth_item = set(row[0] for row in truth_cache)
        intersect = sorted(crowd_item.intersection(truth_item))
        item2idx = dict(zip(intersect, range(len(intersect))))
        crowd2idx = dict(zip(sorted(crowd_worker), range(len(crowd_worker))))
        nsample, dim , nvocab = len(item2idx), len(crowd2idx), len(crowd_response)
        data = np.zeros((nsample, dim, nvocab), dtype=np.int)
        for j, i, l in crowd_cache:
            if j in item2idx and i in crowd2idx:
                data[item2idx[j], crowd2idx[i], l] = 1
        if np.any(data.sum(axis=2) > 1):
            raise ValueError
        if np.any(data.sum(axis=2) == 0):
            logger.info('Data has missing values. A new label is created to represent the missing values.')
            self.is_missing_value = True
            data = np.concatenate((data, 1-data.sum(axis=2, keepdims=True)), axis=2)
        labels = np.zeros(nsample, dtype=np.int)
        for j, l in truth_cache:
            if j in item2idx:
                labels[item2idx[j]] = l
        self.data, self.labels = data, labels

    @staticmethod
    def read_data(file):
        """Read crowdsourcing data"""
        with open(file, 'r') as h:
            cache = []
            j_max, i_max, label_max = 0, 0, 0
            for line in h:
                j, i, label = map(lambda x: int(x) - 1, line.strip().split('\t'))
                j_max = max(j_max, j + 1)
                i_max = max(i_max, i + 1)
                label_max = max(label_max, label + 1)
                cache.append((j, i, label))
        return cache

    @staticmethod
    def read_label(file):
        with open(file, 'r') as h:
            cache = []
            j_max, l_max = 0, 0
            for line in h:
                j, l = map(lambda x: int(x) - 1, line.strip().split('\t'))
                j_max = max(j_max, j + 1)
                l_max = max(l_max, l + 1)
                cache.append((j, l))
        return cache

class MNIST_Reader():
    def __init__(self, folder, binarized=True, threshold=0.5):
        self.mnist = input_data.read_data_sets(folder, one_hot=True)
        # training data
        train = self.mnist.train.images
        if binarized:
            train = np.array(train > threshold, dtype=np.int)[..., np.newaxis]
            self.train = np.concatenate([train, 1 - train], axis=2)
        self.labels = self.mnist.train.labels