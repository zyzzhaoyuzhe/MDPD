"""
Edge data from VanHateren
"""

import time
import scipy.io as scio
from MDPD import *
import numpy as np
from multiprocessing import Pool
import os
import sys
from random import shuffle

# get file list
folder = '/media/vincent/Data/Dataset/VanHateren/data_mat/'
files = []
for item in os.listdir(folder):
    if item.endswith('.mat'):
        files.append(folder + item)

# randomize file list
nb_file = len(files)
# shuffle(files)
# read data
size = 5 # decide the patch size
halfsize = (size-1)/2

n_per_file = 100
def helper(filename, n_per):
    mat = scio.loadmat(filename)
    mat = mat['data_part']
    mat = np.asarray(mat)>0.07
    mat = mat.astype(float)
    patchsize = mat.shape[0]
    center = (patchsize + 1)/2
    idx = range(-halfsize+center, halfsize+center+1)
    mat = mat[idx, :, :, :]
    mat = mat[:, idx, :, :]

    total = mat.shape[3]
    m = mat.shape[0] * mat.shape[1] * mat.shape[2]
    mat = np.transpose(mat.reshape((m, total), order='F'))
    n = n_per if n_per < total else total
    idx = np.random.randint(total, size=n)
    mat = mat[idx, :]
    output = []
    for i in range(m):
        foo = mat[:, i].reshape((1, n))
        output.append(np.append(foo,1-foo,axis=0))
    return output

data = helper(files.pop(), n_per_file)
m = len(data)
for filename in files:
    foo = helper(filename, n_per_file)
    for i in range(m):
        data[i] = np.append(data[i], foo[i], axis=1)

model = MDPD.StageEM()
model.c = 6
model.read(data)
model.learn(stats_output=True, stopcrit='number of iterations', num_iter=100)
model.save('model_edge')

print 1


