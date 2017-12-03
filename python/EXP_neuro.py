from __future__ import division

import os
import numpy as np
import time
import signal
import scipy.io as scio
from scipy.sparse import coo_matrix
from MDPD import *
import matplotlib.pyplot as plt
import matplotlib
import logging

logger = logging.getLogger('EXP_neuro')



data_bool = np.load('../neuroscience_dataset/fluordata1_bin.npy')
dim, ntime = data_bool.shape

# convert data to the format used by MDPD
data = np.zeros([dim, ntime, 2], dtype=np.int)
data[data_bool, 0] = 1
data[np.logical_not(data_bool), 1] = 1

#
window = 100

images = []
for frame in xrange(ntime - window):
    img = utils.MI_data(data[:, frame:frame+window, :]).sum(axis=(1, 3))
    images.append(img)