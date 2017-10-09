"""
Learning Mixture model from mouse data. And save the model.
"""

import time
import scipy.io as scio
import matplotlib.pyplot as plt
from MDPD import *
import numpy as np
from multiprocessing import Pool
import sys

option = sys.argv[1]
# option = 'egr'


folder = '/home/vincent/Dropbox/Research/Project/Stryker_Mouse/'
filename = folder + 'data_151210data003.mat'
inputs = [('data_151210data003.mat', 'model_151210data003_', option),
          ('data_151211data007.mat', 'model_151211data007_', option),
          ('data_151220data003.mat', 'model_151220data003_', option)]

def mat2MDPD(fn):
    mat = scio.loadmat(fn)

    data_g = mat['data_g']
    label_g = mat['label_g']
    data_n = mat['data_n']
    label_n = mat['label_n']
    v_g = mat['v_g']
    v_n = mat['v_n']
    celltype = mat['celllabel']

    return data_g, label_g, data_n, label_n, v_g, v_n, celltype

def main(inputs):
    starttime = time.time()
    option = inputs[2]
    filenamein = folder + inputs[0]
    filenameout = folder + inputs[1] + option
    data_g, label_g, data_n, label_n, v_g, v_n, celltype = mat2MDPD(filenamein)

    if 'e' in option:
        isex = 1
    elif 'i' in option:
        isex = 0
    else:
        isex = 2
    if 'g' in option:
        isgn = 1
    elif 'n' in option:
        isgn = 0
    else:
        sys.stderr('error: pick grating or non grating')
        return
    if 'r' in option:
        isrun = 1
    elif 's' in option:
        isrun = 0
    else:
        isrun = 2

    ## prepare data
    celltype = np.asarray(celltype).squeeze()
    # grating or non grating
    if isgn == 1:
        data = data_g
        label = label_g
    elif isgn == 0:
        data = data_n
        label = label_n

    # excitatory or inhibitory
    foo = np.asarray(data)
    if isex == 1:
        foo = foo[:, :, celltype == 1]  # include only excitatory neurons.
    elif isex == 0:
        foo = foo[:, :, celltype == 1]  # include only inhibitory neurons.
    else:
        pass    # include all neurons
    label = np.asarray(label)

    # run or stay
    thres = 6
    if isgn == 1:   # grating
        vlabel = np.asarray(v_g).squeeze()
        if isrun == 1:  # run state
            mask = vlabel > thres
            foo = foo[:, mask, :]
            label = label[mask]
        elif isrun == 0: # stay state
            mask = vlabel <= thres
            foo = foo[:, mask, :]
            label = label[mask]
        else:   # both
            pass
    elif isgn == 0: # non-grating
        vlabel = np.asarray(v_n).squeeze()
        if isrun == 1:  # run state
            mask = vlabel > thres
            foo = foo[:, mask, :]
            label = label[mask]
        elif isrun == 0:  # stay state
            mask = vlabel <= thres
            foo = foo[:, mask, :]
            label = label[mask]
        else:  # both
            pass
    # body
    z = []
    m = foo.shape[2]
    for i in range(m):
        z.append(foo[:, :, i])

    #
    model = MDPD.StageEM()
    model.read(z, label=label)
    model.c = 6

    model.learn(stopcrit='number of iterations', stats_output=True)
    model.save(filenameout)
    print time.time()-starttime

if __name__ == "__main__":
    # main(inputs[0])
    p = Pool(3)
    p.map(main, inputs)

