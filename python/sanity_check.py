import time
import scipy.io as scio
import numpy as np
from multiprocessing import Pool
import sys
from copy import copy
from scipy import stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from MDPD import utils, readers, MDPD
from MDPD.readers import *
import os


folder = '/media/vzhao/Data/crowdsourcing_datasets/'
# folder = '/Users/vincent/Documents/Research/MDPD/crowdsourcing_datasets'

reader = Crowd_Sourcing_Readers(os.path.join(folder, 'bird', 'bluebird_crowd.txt'), os.path.join(folder, 'bird', 'bluebird_truth.txt'))
train, label = reader.data, reader.labels
lock = np.zeros(train.shape[1:], dtype=np.bool)
print train.shape

NSAMPLE, DIM, NVOCAB = train.shape
EFF_NVOCAB = NVOCAB-1 if reader.is_missing_value else NVOCAB

##
pick = range(20)
sample_log_weights = np.ones(NSAMPLE) * -10000
sample_log_weights[pick] = 1
sample_log_weights = sample_log_weights - logsumexp(sample_log_weights)

score = utils.Feature_Selection.MI_score(train, sample_log_weights=sample_log_weights, rm_diag=True)
sigma1 = score.sum(axis=1) / (DIM-1)

##
score = utils.Feature_Selection.MI_score(train[pick, ...] , rm_diag=True)
sigma2 = score.sum(axis=1) / (DIM-1)


pass