import time
import scipy.io as scio
import numpy as np
from multiprocessing import Pool
import sys
from copy import copy
import matplotlib.pyplot as plt
from MDPD import utils, readers, MDPD


# folder = "/media/vzhao/Data/MNIST"
folder = "/Users/vincent/Documents/Research/MDPD/MNIST"
mnist = readers.MNIST_Reader(folder, binarized=True)
train, labels = mnist.train, mnist.labels
_, dim, _ = train.shape
# data per digit
train_uni = [None] * 10
for dig in xrange(10):
    train_uni[dig] = train[labels==dig,...]
# small sample
train_small = train[:5000,...]
labels_small = labels[:5000]

data, labs = train_small, labels_small

#########################
Ntop = 200
model = MDPD.MDPD_online()
model.fit(data, ncomp=10,
          features=Ntop, init='random',
          epoch=50, batch=500, update_features_per_batch=10,
          verbose=False)




# MDPD model

model = MDPD.MDPD2()

model.fit(train, 10, 200)






print 'END'

# nsample, d1, d2 = X_train.shape
# dim = d1*d2
# ncomp = 10
# nvocab = 2
#
# data = X_train.reshape((nsample, dim))>128
# # plt.imshow(data[1,:].squeeze().reshape(28,28))
# # plt.show()
# data = data.T[:, np.newaxis, :]
# data = np.concatenate((data, 1-data), axis=1)
# data = np.einsum(data, [1, 2, 0])
# data = data[y_train==3, :, :]
# nsample, dim, nvocab = data.shape
#
# test = X_test.reshape((X_test.shape[0], dim))>128
# # plt.imshow(data[1,:].squeeze().reshape(28,28))
# # plt.show()
# test = test.T[:, np.newaxis, :]
# test = np.concatenate((test, 1-test), axis=1)
# test = np.einsum(test, [1, 2, 0])
#
#
#
# model = MDPD.MDPD()
# model.get_config(dim=dim, nsample=nsample, ncomp=10, nvocab=nvocab)
# model.reset(data)
# MI_origin = model.get_MIres(data, rm_diag=False)
# model = MDPD.MDPD()
# model.get_config(dim=dim, nsample=nsample, ncomp=50, nvocab=nvocab)
#
#
# output = model.fit(data, init="StageEM", stopcrit='niter', niter=100)
# MI_remain = model.get_MI(data, rm_diag=True)
#
#
# plt.imshow(model.C[:, 0, 12].reshape(28, 28)); plt.show()

