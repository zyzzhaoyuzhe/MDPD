import time
import scipy.io as scio
from MDPD import *
import numpy as np
from multiprocessing import Pool
import sys
from keras.datasets import mnist
from copy import copy
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

nsample, d1, d2 = X_train.shape
dim = d1*d2
ncomp = 10
nvocab = 2

data = X_train.reshape((nsample, dim))>128
# plt.imshow(data[1,:].squeeze().reshape(28,28))
# plt.show()
data = data.T[:, np.newaxis, :]
data = np.concatenate((data, 1-data), axis=1)
data = np.einsum(data, [1, 2, 0])
data = data[y_train==3, :, :]
nsample, dim, nvocab = data.shape

test = X_test.reshape((X_test.shape[0], dim))>128
# plt.imshow(data[1,:].squeeze().reshape(28,28))
# plt.show()
test = test.T[:, np.newaxis, :]
test = np.concatenate((test, 1-test), axis=1)
test = np.einsum(test, [1, 2, 0])



model = MDPD.MDPD()
model.get_config(dim=dim, nsample=nsample, ncomp=10, nvocab=nvocab)
model.reset(data)
MI_origin = model.get_MIres(data, rm_diag=False)
model = MDPD.MDPD()
model.get_config(dim=dim, nsample=nsample, ncomp=50, nvocab=nvocab)


output = model.train(data, method="StageEM", stopcrit='niter', niter=100)
MI_remain = model.get_MI(data, rm_diag=True)


plt.imshow(model.C[:, 0, 12].reshape(28, 28)); plt.show()

