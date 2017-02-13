"""
A module that helps MDPD module
"""


import numpy as np
import sys
from scipy.misc import logsumexp
from copy import deepcopy


def log_comL(nsample, data, W, C):
    """
    Calculate log of the joint distribution for each data and each component. The size of the output is c by n.
    :param nsample:
    :param data:
    :param W:
    :param C:
    :return:
    """
    dim = len(data)
    ncomp = np.size(W)
    if dim > 0:
        out = np.zeros((ncomp, nsample))
        # log(f(x|y))
        for i in range(dim):
            out += MyLog(np.dot(C[i].transpose(), data[i]))
        # log(f(y))
        logW = MyLog(W).reshape((ncomp,1))
        out += logW
    else:
        out = np.zeros((ncomp, nsample))
    return out


def log_comL_fast(nsample, data, logW, logC):
    if data.any() and logC.any():
        foo = np.tensordot(logC, data, axes=[(0, 1), (1, 2)])
    else:
        ncomp = logW.size
        foo = np.zeros((ncomp, nsample))
    return foo + logW[:, np.newaxis]


def MyLog(input):
    """

    :param input:
    :return:
    """
    if isinstance(input,int):
        return np.log(input) if input != 0 else np.log(sys.float_info.min)
    else:
        if isinstance(input,list):
            input = np.asarray(input)
        _shape = input.shape
        input = input.flatten()
        _len = np.size(input)
        output = np.zeros(_len)
        for i in range(_len):
            foo = input[i]
            output[i] = np.log(foo) if foo != 0 else np.log(sys.float_info.min)
        return output.reshape(_shape)


def comp_duplicate(logW, logC, comp):
    newlogW = np.copy(logW)
    newlogC = np.copy(logC)
    newlogC = np.append(newlogC, newlogC[:, :, comp][:, :, np.newaxis], axis=2)
    newlogW = np.append(newlogW, newlogW[comp] - np.log(2))
    newlogW[comp] -= np.log(2)
    return newlogW, newlogC, newlogC.shape[2]


def logposterior_StageEM(data, logW, logC, infoset):
    nsample = data.shape[0]
    data_tmp = data[:, infoset, :]
    C_tmp = logC[infoset, :, :]
    foo = log_comL_fast(nsample, data_tmp, logW, C_tmp)
    lsum = logsumexp(foo, axis=0)
    return foo - lsum


def Mstep(logpost, data):
    nsample, dim, nvocab = data.shape
    # update C
    post = np.exp(logpost)
    newlogC = np.log(np.tensordot(data, post, axes=(0, 1))) - logsumexp(logpost, axis=1)
    # update W
    newlogW = logsumexp(logpost, axis=1) - np.log(nsample)
    return newlogW, newlogC


def get_MIres(data, W, C, infoset, rm_diag=False):
    dim = len(data)
    nsample = data[0].shape[1]
    ncomp = C[0].shape[1] if len(C[0].shape)>1 else 1
    # EM based on current model
    logpost = logposterior_StageEM(data, W, C, infoset)
    newW, newC = Mstep(logpost, data)
    ## Body
    MIres = np.empty([dim, dim, ncomp])
    sqrtpost = np.sqrt(logpost)
    for k in range(ncomp):
        for i in range(dim):
            zi_weighted = data[i] * sqrtpost[k, :]
            for j in range(dim):
                zj_weighted = data[j] * sqrtpost[k, :]
                second = 1. / nsample * np.dot(zi_weighted, zj_weighted.transpose())
                second = second/newW[k]
                first = np.outer(newC[i][:, k], newC[j][:, k])
                if rm_diag and i==j:
                    MIres[i, j, k] = 0
                else:
                    MIres[i, j, k] = np.sum(second * MyLog(div0(second, first)))
    return MIres


def get_MIres_fast(data, logW, logC, infoset, rm_diag=False, wPMI=False):
    nsample, dim, nvocab = data.shape
    ncomp = logC[0].shape[1]
    logpost = logposterior_StageEM(data, logW, logC, infoset)
    newlogW, newlogC = Mstep(logpost, data)
    post = np.exp(logpost)
    foo_data = data[:, :, :, np.newaxis] * np.sqrt(post).T[:, np.newaxis, np.newaxis, :]
    output = np.zeros((dim, dim, ncomp))
    for k in range(ncomp):
        second = 1. / nsample * np.tensordot(foo_data[:, :, :, k], foo_data[:, :, :, k], axes=(0, 0))
        second = second / np.exp(newlogW[k])
        logfirst = np.add.outer(newlogC[:, :, k], newlogC[:, :, k])
        foo = second * (np.log(second) - logfirst)
        foo[logfirst == 0] = 0
        foo[second == 0] = 0
        foo = foo.sum(axis=(1, 3))
        output[:, :, k] = foo
        if rm_diag:
            np.fill_diagonal(output[:, :, k], 0)
    if wPMI:
        return output * np.exp(newlogW[np.newaxis, np.newaxis, :])
    else:
        return output


def get_MI(data, W, C, infoset, rm_diag=False):
    logpost = logposterior_StageEM(data, W, C, infoset)
    newlogW, newlogC = Mstep(logpost, data)
    MIres = get_MIres_fast(data, newlogW, newlogC, infoset, rm_diag=rm_diag, wPMI=True)
    return np.sum(MIres,axis=2)

def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c

def get_Hessian(data, W, C, lock, infoset, cord1, cord2, comp):
    """

    """
    # duplicate
    nsample, dim, nvocab = data.shape
    W, C, ncomp = comp_duplicate(W, C, comp)
    # add cord1 and cord2 into activeset
    infoset = add_infoset(infoset, cord1)
    infoset = add_infoset(infoset, cord2)
    lock1 = lock[cord1, :, comp]
    lock2 = lock[cord2, :, comp]
    free1 = np.where(lock1==0)[0]
    r1 = free1.size
    free2 = np.where(lock2==0)[0] + r1
    r2 = free2.size
    indices = np.concatenate((free1[:-1], free2[:-1]))

    # old MI
    oldMI = get_MI(data, W, C, infoset)[cord1, cord2]
    # perturb C
    eta = 0.001
    H = np.zeros((r1 + r2 - 2, r1 + r2 - 2))
    for p, idx1 in enumerate(indices):
        for q, idx2 in enumerate(indices):
            dmu = np.zeros(2*nvocab)
            dmu[idx1] += eta
            dmu[idx2] += eta
            dmu_norm = np.linalg.norm(dmu)

            dmu[free1[-1]] = -np.sum(dmu[:nvocab - 1])
            dmu[free2[-1]] = -np.sum(dmu[nvocab:(2*nvocab-1)])

            dmu1 = dmu[:nvocab]
            dmu2 = dmu[nvocab:]

            newC1 = np.copy(C)
            newC1[cord1, :, comp] += dmu1
            newC1[cord1, :, -1] -= dmu1
            newC1[cord2, :, comp] += dmu2
            newC1[cord2, :, -1] -= dmu2

            newC2 = np.copy(C)
            newC2[cord1, :, comp] -= dmu1
            newC2[cord1, :, -1] += dmu1
            newC2[cord2, :, comp] -= dmu2
            newC2[cord2, :, -1] += dmu2

            newMI1 = get_MI(data, W, newC1, infoset)[cord1, cord2]
            newMI2 = get_MI(data, W, newC2, infoset)[cord1, cord2]

            H[p, q] = (newMI1 + newMI2 - 2 * oldMI) / (dmu_norm ** 2)
    return H

def get_Hessian2(data, logW, logC, lock, infoset, cord1, cord2, comp):
    '''Return hessian with regard to cord1 and cord2'''
    nsample, dim, nvocab = data.shape
    idx1 = np.nonzero(lock[cord1])[0].tolist()
    idx2 = np.nonzero(lock[cord2])[0].tolist()
    sample_part = [[[] for i in xrange(len(idx2))] for j in xrange(len(idx1))]
    output = np.zeros([len(idx1), len(idx2)])
    for n in xrange(nsample):
        p = np.where(data[n, cord1, :])[0][0]
        q = np.where(data[n, cord2, :])[0][0]
        if p in idx1 and q in idx2:
            sample_part[idx1.index(p)][idx2.index(q)].append(n)

    for i in xrange(len(idx1)):
        for j in xrange(len(idx2)):
            foo = np.tensordot(logC, data[sample_part[i][j], :, :], axes=[(0, 1), (1, 2)])
            foo += logW[:, np.newaxis]
            output[i, j] = np.exp(foo[:, comp] - logC[cord1, idx1[i], comp]
                                  - logC[cord2, idx2[j], comp] - logsumexp(foo, axis=1))
    return output

def add_infoset(activeset, inputs):
    if isinstance(inputs, int):
        inputs = [inputs]
    for item in inputs:
        if item not in activeset:
            activeset.append(item)
    return activeset


### Stop criterion
## likelihood
def stopcrit_likelihood(current, train_config):
    return current['newL'] - current['oldL'] < train_config['diffthre'] or current['count'] > train_config['maxiter']


## mutual information
def stopcrit_MI(current, train_config):
    return abs(current['newMI'] - current['oldMI']) < train_config['diffthre'] or current['count'] > train_config['maxiter']


## number of components
def stopcrit_num_component(current, train_config):
    return current.c >= train_config['num_component']


## number of iterations
def stopcrit_num_iter(current, train_config):
    return current['count'] >= train_config['maxiter']
