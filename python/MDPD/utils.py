"""
A module that helps MDPD module
"""

from __future__ import division
import numpy as np
import sys
from scipy.misc import logsumexp
from copy import deepcopy
import tensor_power as tp

import time


# def log_comL(nsample, data, W, C):
#     """
#     Calculate log of the joint distribution for each data and each component. The size of the output is c by n.
#     :param nsample:
#     :param data:
#     :param W:
#     :param C:
#     :return:
#     """
#     dim = len(data)
#     ncomp = np.size(W)
#     if dim > 0:
#         out = np.zeros((ncomp, nsample))
#         # log(f(x|y))
#         for i in range(dim):
#             out += MyLog(np.dot(C[i].transpose(), data[i]))
#         # log(f(y))
#         logW = MyLog(W).reshape((ncomp,1))
#         out += logW
#     else:
#         out = np.zeros((ncomp, nsample))
#     return out

#####################################3
# Inference
def log_joint_prob_fast(data, logW, logC):
    """
    Log joint probability
    :param nsample:
    :param data:
    :param logW:
    :param logC:
    :return: c-d
    """
    if data.any() and logC.any():
        foo = np.tensordot(data, logC, axes=[(1, 2), (0, 1)])
    else:
        nsample, ncomp = data.shape[0], logW.size
        foo = np.zeros((ncomp, nsample))
    return foo + logW[np.newaxis, :]


def mstep(logpost, data):
    """

    :param logpost:  n-c
    :param data:  n-d-r
    :return:
    """
    nsample, dim, nvocab = data.shape

    # a better implementation
    newlogC = logsumexp(logpost[:, np.newaxis, np.newaxis, :], axis=0, b=data[..., np.newaxis]) \
              - logsumexp(logpost, axis=0)[np.newaxis, np.newaxis, :]
    # old implementation
    # post = np.exp(logpost)
    # newlogC = np.log(np.tensordot(data, post, axes=(0, 0))) - logsumexp(logpost, axis=0)[np.newaxis, np.newaxis, :]
    # newlogC[np.isinf(newlogC)] = -100

    newlogC -= logsumexp(newlogC, axis=1)[:, np.newaxis, :]
    # update W
    newlogW = logsumexp(logpost, axis=0) - np.log(nsample)
    return newlogW, newlogC


#####################################3
# MDPD initialization
def init_mv(data, feature_set):  # majority vote
    data_selected = data[:, feature_set, :]
    votes = data_selected.sum(axis=1)  # n-r
    logpost = np.log(votes / votes.sum(axis=1)[:, np.newaxis])
    logpost[np.isneginf(logpost)] = -100
    return mstep(logpost, data_selected)


def init_random(dim, ncomp, nvocab):
    logW = np.ones(ncomp) / ncomp
    logC = np.random.random_sample((dim, nvocab, ncomp))
    logC /= logC.sum(axis=1)[:, np.newaxis, :]
    return logW, logC


def init_spectral(data, ncomp):
    """
            Use spectral methods to initialize EM. (Zhang. and Jordan. 2014)
            :param data:
            :return:
            """
    nsample, dim, nvocab = data.shape
    # divide workers into 3 partitions
    num_worker_partition = int(dim / 3)
    # np.random.seed(seed=10)
    foo = np.random.permutation(dim)

    partition = [None] * 3
    partition[0] = foo[:num_worker_partition]
    partition[1] = foo[num_worker_partition:2 * num_worker_partition]
    partition[2] = foo[2 * num_worker_partition:]

    # calculate average response to each sample for each group
    train_g = [None] * 3
    for g in range(3):
        foo = data[:, partition[g], :]
        train_g[g] = np.mean(foo, axis=1).T

    #
    perm = [[1, 2, 0], [2, 0, 1], [0, 1, 2]]
    #
    W = []
    Cg = []
    for g in range(3):
        a = perm[g][0]
        b = perm[g][1]
        c = perm[g][2]
        m2, m3 = tp.get_tensor(train_g, a, b, c)
        foo, bar = tp.tensorpower(m2, m3, ncomp)
        W.append(foo)
        Cg.append(bar)
    W = np.mean(np.asarray(W), axis=0)
    # normalize W
    W /= sum(W)
    # use Cg to recover C for each worker
    C = np.zeros((dim, nvocab, ncomp))
    for i in range(dim):
        foo = np.zeros((ncomp, ncomp))
        for g in range(3):
            if i not in partition[g]:
                foo += tp.get_Ci(train_g, W, Cg, g, data, i)
        foo = foo / 2
        C[i, :, :] = foo
        # C.append(foo)
    return np.log(W), np.log(C)


#####################################
# MDPD feature selection
class Feature_Selection():
    @classmethod
    def MI_feature_selection(cls, data, topN):
        ranking, mi_score = cls.MI_feature_ranking(data)
        return ranking[:topN], mi_score[ranking]

    @classmethod
    def MI_feature_ranking(cls, data):
        score = cls.MI_score(data, rm_diag=True)
        # mi_score = pairwise_MI.sum(axis=(1, 3))
        # np.fill_diagonal(mi_score, 0)
        mi_score = score.sum(axis=1)
        ranking = np.argsort(mi_score, axis=None)[::-1]
        return ranking, mi_score

    @classmethod
    def MI_score(cls, data, rm_diag=False):
        nsample, dim, nvocab = data.shape
        logpost = np.zeros([nsample, 1])
        newlogW, newlogC = mstep(logpost, data)
        second = 1. / nsample * np.tensordot(data, data, axes=(0, 0))
        logfirst = np.add.outer(newlogC[:, :, 0], newlogC[:, :, 0])
        pmi = second * (np.log(second) - logfirst)
        pmi[logfirst == 0] = 0
        pmi[second == 0] = 0
        score = pmi.sum(axis=(1, 3))
        if rm_diag:
            np.fill_diagonal(score, 0)
        return score

    @classmethod
    def MI_score_conditional(cls, data, logpost, rm_diag=False):
        """

        :param data:
        :param logpost:
        :param rm_diag:
        :return: d-d-c, c
        """
        nsample, dim, nvocab = data.shape
        ncomp = logpost.shape[1]
        newlogW, newlogC = mstep(logpost, data)
        post = np.exp(logpost)
        foo_data = data[:, :, :, np.newaxis] * np.sqrt(post)[:, np.newaxis, np.newaxis, :]
        score = np.zeros((dim, dim, ncomp))
        for k in range(ncomp):
            second = 1. / nsample * np.tensordot(foo_data[:, :, :, k], foo_data[:, :, :, k], axes=(0, 0))
            second = second / np.exp(newlogW[k])
            logfirst = np.add.outer(newlogC[:, :, k], newlogC[:, :, k])
            pmi = second * (np.log(second) - logfirst)
            pmi[logfirst == 0] = 0
            pmi[second == 0] = 0
            score[:, :, k] = pmi.sum(axis=(1, 3))
            if rm_diag:
                np.fill_diagonal(score[:, :, k], 0)
        return score, np.exp(newlogW)

    @classmethod
    def MI_score_conditional_faster(cls, data, logpost, rm_diag=False):
        nsample, dim, nvocab = data.shape
        ncomp = logpost.shape[1]
        newlogW, newlogC = mstep(logpost, data)
        data_out = data[..., np.newaxis, np.newaxis, np.newaxis] * data[:, np.newaxis, np.newaxis, :, :, np.newaxis]
        logpost_reshape = logpost[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
        log_first = newlogC[:, :, np.newaxis, np.newaxis, :] + newlogC[np.newaxis, np.newaxis, ...]
        log_second = logsumexp(logpost_reshape, axis=0, b=data_out) - np.log(nsample) - np.reshape(newlogW,
                                                                                                   (1, 1, 1, 1, -1))
        pmi = np.exp(log_second) * (log_second - log_first)
        pmi[np.isinf(log_second)] = 0
        score = np.sum(pmi, axis=(1, 3))
        for k in xrange(ncomp):
            if rm_diag:
                np.fill_diagonal(score[..., k], 0)
        return score, np.exp(newlogW)


# Mutual Information in the data conditional on the model
# def get_MIres(data, W, C, infoset, rm_diag=False):
#     dim = len(data)
#     nsample = data[0].shape[1]
#     ncomp = C[0].shape[1] if len(C[0].shape) > 1 else 1
#     # EM based on current model
#     logpost = logposterior_StageEM(data, W, C, infoset)
#     newW, newC = mstep(logpost, data)
#     ## Body
#     MIres = np.empty([dim, dim, ncomp])
#     sqrtpost = np.sqrt(logpost)
#     for k in range(ncomp):
#         for i in range(dim):
#             zi_weighted = data[i] * sqrtpost[k, :]
#             for j in range(dim):
#                 zj_weighted = data[j] * sqrtpost[k, :]
#                 second = 1. / nsample * np.dot(zi_weighted, zj_weighted.transpose())
#                 second = second / newW[k]
#                 first = np.outer(newC[i][:, k], newC[j][:, k])
#                 if rm_diag and i == j:
#                     MIres[i, j, k] = 0
#                 else:
#                     MIres[i, j, k] = np.sum(second * mylog(div0(second, first)))
#     return MIres


# def get_MI(data, W, C, infoset, rm_diag=False):
#     logpost = logposterior_StageEM(data, W, C, infoset)
#     newlogW, newlogC = mstep(logpost, data)
#     MIres = MIres_fast(data, newlogW, newlogC, infoset, rm_diag=rm_diag, weighted=True)
#     return np.sum(MIres, axis=2)


def mylog(input):
    """

    :param input:
    :return:
    """
    if isinstance(input, int):
        return np.log(input) if input != 0 else np.log(sys.float_info.min)
    else:
        if isinstance(input, list):
            input = np.asarray(input)
        _shape = input.shape
        input = input.flatten()
        _len = np.size(input)
        output = np.zeros(_len)
        for i in range(_len):
            foo = input[i]
            output[i] = np.log(foo) if foo != 0 else np.log(sys.float_info.min)
        return output.reshape(_shape)

# def comp_duplicate(logW, logC, lock, comp):
#     newlogW = np.copy(logW)
#     newlogW = np.append(newlogW, newlogW[comp] - np.log(2))
#     newlogW[comp] -= np.log(2)
#
#     newlogC = np.copy(logC)
#     newlogC = np.append(newlogC, newlogC[:, :, comp][:, :, np.newaxis], axis=2)
#
#     lock = np.append(lock, lock[:, :, comp][:, :, np.newaxis], axis=2)
#     return newlogW, newlogC, lock, newlogC.shape[2]


# def logposterior_StageEM(data, logW, logC, infoset):
#     nsample = data.shape[0]
#     data_tmp = data[:, infoset, :]
#     C_tmp = logC[infoset, :, :]
#     foo = log_joint_prob_fast(nsample, data_tmp, logW, C_tmp)
#     lsum = logsumexp(foo, axis=0)
#     return foo - lsum


#
#
# def div0(a, b):
#     """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
#     with np.errstate(divide='ignore', invalid='ignore'):
#         c = np.true_divide(a, b)
#         c[~np.isfinite(c)] = 0  # -inf inf NaN
#     return c
#
#
# def get_Hessian(data, W, C, lock, infoset, cord1, cord2, comp):
#     """
#
#     """
#     # duplicate
#     nsample, dim, nvocab = data.shape
#     W, C, ncomp = comp_duplicate(W, C, comp)
#     # add cord1 and cord2 into activeset
#     infoset = add_infoset(infoset, cord1)
#     infoset = add_infoset(infoset, cord2)
#     lock1 = lock[cord1, :, comp]
#     lock2 = lock[cord2, :, comp]
#     free1 = np.where(lock1 == 0)[0]
#     r1 = free1.size
#     free2 = np.where(lock2 == 0)[0] + r1
#     r2 = free2.size
#     indices = np.concatenate((free1[:-1], free2[:-1]))
#
#     # old MI
#     oldMI = get_MI(data, W, C, infoset)[cord1, cord2]
#     # perturb C
#     eta = 0.001
#     H = np.zeros((r1 + r2 - 2, r1 + r2 - 2))
#     for p, idx1 in enumerate(indices):
#         for q, idx2 in enumerate(indices):
#             dmu = np.zeros(2 * nvocab)
#             dmu[idx1] += eta
#             dmu[idx2] += eta
#             dmu_norm = np.linalg.norm(dmu)
#
#             dmu[free1[-1]] = -np.sum(dmu[:nvocab - 1])
#             dmu[free2[-1]] = -np.sum(dmu[nvocab:(2 * nvocab - 1)])
#
#             dmu1 = dmu[:nvocab]
#             dmu2 = dmu[nvocab:]
#
#             newC1 = np.copy(C)
#             newC1[cord1, :, comp] += dmu1
#             newC1[cord1, :, -1] -= dmu1
#             newC1[cord2, :, comp] += dmu2
#             newC1[cord2, :, -1] -= dmu2
#
#             newC2 = np.copy(C)
#             newC2[cord1, :, comp] -= dmu1
#             newC2[cord1, :, -1] += dmu1
#             newC2[cord2, :, comp] -= dmu2
#             newC2[cord2, :, -1] += dmu2
#
#             newMI1 = get_MI(data, W, newC1, infoset)[cord1, cord2]
#             newMI2 = get_MI(data, W, newC2, infoset)[cord1, cord2]
#
#             H[p, q] = (newMI1 + newMI2 - 2 * oldMI) / (dmu_norm ** 2)
#     return H
#
#
# def get_Hessian2(data, logW, logC, lock, infoset, cord1, cord2, comp):
#     '''Return hessian with regard to cord1 and cord2'''
#     nsample, dim, nvocab = data.shape
#     idx1 = np.nonzero(lock[cord1, :, comp])[0].tolist()
#     idx2 = np.nonzero(lock[cord2, :, comp])[0].tolist()
#     sample_part = [[[] for i in xrange(len(idx2))] for j in xrange(len(idx1))]
#     output = np.zeros([len(idx1), len(idx2)])
#     for n in xrange(nsample):
#         p = np.where(data[n, cord1, :])[0][0]
#         q = np.where(data[n, cord2, :])[0][0]
#         if p in idx1 and q in idx2:
#             sample_part[idx1.index(p)][idx2.index(q)].append(n)
#
#     for i in xrange(len(idx1)):
#         for j in xrange(len(idx2)):
#             foo = np.tensordot(logC, data[sample_part[i][j], :, :], axes=[(0, 1), (1, 2)])
#             foo += logW[:, np.newaxis]
#             # foo.shape = (ncomp, nsample)
#             output[i, j] = 1 / nsample * np.sum(np.exp(foo[comp, :] - logC[cord1, idx1[i], comp]
#                                                        - logC[cord2, idx2[j], comp] - logsumexp(foo, axis=0)))
#     return output
#
#
#
# ### Stop criterion
# ## likelihood
# def stopcrit_likelihood(current, train_config):
#     return current['newL'] - current['oldL'] < train_config['diffthre'] or current['count'] > train_config['maxiter']
#
#
# ## mutual information
# def stopcrit_MI(current, train_config):
#     return abs(current['newMI'] - current['oldMI']) < train_config['diffthre'] or current['count'] > train_config[
#         'maxiter']
#
#
# ## number of components
# def stopcrit_num_component(current, train_config):
#     return current.c >= train_config['num_component']
#
#
# ## number of iterations
# def stopcrit_num_iter(current, train_config):
#     return current['count'] >= train_config['maxiter']
