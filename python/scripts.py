from __future__ import division

import os
import numpy as np
import time, timeit
import signal
import scipy.io as scio
from scipy import stats
from scipy.sparse import coo_matrix
from MDPD.readers import *
from MDPD import utils, MDPD
import matplotlib.pyplot as plt
import matplotlib
import pickle

folder = '/media/vzhao/Data/crowdsourcing_datasets/'
# folder = '/Users/vincent/Documents/Research/MDPD/crowdsourcing_datasets'

reader = Crowd_Sourcing_Readers(os.path.join(folder, 'web', 'web_crowd.txt'), os.path.join(folder, 'web', 'web_truth.txt'))
train, label = reader.data, reader.labels
lock = np.zeros(train.shape[1:],dtype=np.bool)
lock[:, -1] = 1
print(train.shape)

NSAMPLE, DIM, NVOCAB = train.shape
EFF_NVOCAB = NVOCAB-1 if reader.is_missing_value else NVOCAB


score = utils.Feature_Selection.MI_score(train, lock=lock)
sigma = score.sum(axis=1)
features = np.argsort(sigma)[::-1]
sigma = sigma[features]
# features, sigma = utils.Feature_Selection.MI_feature_ranking(train, lock=lock)
print(features)


# Ntop = 10
#
# mask = train[:, features[:Ntop], -1].sum(axis=1) != Ntop
# train_valid, label_valid = train[mask, :, :], label[mask]
# item_to_guess = len(label) - mask.sum()
#
# # Feature Selection
# model = MDPD.MDPD_standard()
# model.fit(train, EFF_NVOCAB, features=features[:Ntop], init='spectral', verbose=False, epoch=50, lock=lock)
# total_right = model.accuracy(train_valid, label_valid) * len(label_valid) + item_to_guess / EFF_NVOCAB
# print(1 - total_right / len(label))
# model.MI_residue(train)


x_range = range(6, 178)
mv_err = []
mvem_err = []
spec_err = []
for Ntop in x_range:
    print(Ntop)
    # solve if no training data on one item
    mask = train[:, features[:Ntop], -1].sum(axis=1) != Ntop
    train_valid, label_valid = train[mask, :, :], label[mask]
    item_to_guess = len(label) - mask.sum()

    # mv
    # votes = train_valid[:, features[:Ntop], :-1].sum(axis=1)
    # pred = np.argmax(votes, axis=1)
    # total_right = (pred - label_valid == 0).sum() + item_to_guess / EFF_NVOCAB
    # mv_err.append(1 - total_right / len(label))

    # # mvem
    # model = MDPD.MDPD_standard()
    # model.fit(train, EFF_NVOCAB, features=features[:Ntop], init='majority', verbose=False, epoch=50, lock=lock)
    # total_right = model.accuracy(train_valid, label_valid) * len(label_valid) + item_to_guess / EFF_NVOCAB
    # mvem_err.append(1 - total_right / len(label))

    # spec
    cache = []
    for _ in range(10):
        try:
            model = MDPD.MDPD_standard()
            model.fit(train, EFF_NVOCAB, features=features[:Ntop], init='spectral', verbose=False, epoch=50, lock=lock)
            total_right = model.accuracy(train_valid, label_valid) * len(label_valid) + item_to_guess / EFF_NVOCAB
            cache.append(1 - total_right / len(label))
        except:
            pass
    spec_err.append(cache)


with open('NIPS_2018_WEB_SPECEM.p', 'wb') as f:
    pickle.dump(spec_err, f)
pass