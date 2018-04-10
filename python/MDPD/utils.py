"""
A module that helps MDPD module
"""

from __future__ import division
import numpy as np
import sys
# from scipy.special import logsumexp
from copy import deepcopy
import tensor_power as tp

import time

NINF = np.finfo('f').min
NINF_ = -float('inf')
PESP = np.finfo('f').eps


################# Inference ####################
def log_joint_prob_fast(data, logW, logC):
    """
    Log joint probability log(P(X,Y))
    :param nsample:
    :param data: n - d - r
    :param logW: c
    :param logC: d - r - c
    :return: n - c
    """
    if data.any() and logC.any():
        foo = data[..., None] * logC[None, ...]
        # get nan when 0 * -inf
        foo[np.isnan(foo)] = 0
        foo = foo.sum(axis=(1, 2))
        # foo = np.tensordot(data, logC, axes=[(1, 2), (0, 1)])
    else:
        nsample, ncomp = data.shape[0], logW.size
        foo = np.zeros((ncomp, nsample))
    return foo + logW[None, :]


def log_joint_prob_slice(data, logW_slice, logC_slice):
    """
    Log joint Probability log(P(X, Y = c))
    :param data: n - d - r
    :param logW_slice: 1
    :param logC_slice: d - r
    :return: n
    """
    if data.shape[1] == 0:
        return np.zeros(data.shape[0]) + logW_slice
    # make sure no -inf in logC
    foo = data * logC_slice[None, ...]
    foo = foo.sum(axis=(1, 2))
    return foo + logW_slice


def mstep(log_post, data, sample_log_weights=None):
    """

    :param log_post:  n-c
    :param data:  n-d-r
    :return:
    """
    nsample, dim, nvocab = data.shape

    if sample_log_weights is None:
        sample_log_weights = - np.log(nsample) * np.ones(nsample, dtype=np.float)

    log_p_tilde = log_post + sample_log_weights[:, None]    # log(p(y|x_i)p_0(x_i))

    newlogW = logsumexp(log_p_tilde, axis=0)
    newlogW = newlogW - logsumexp(newlogW)

    # NOTE: use logsumexp with arg b might be very slow
    newlogC = logsumexp(log_p_tilde[:, None, None, :], axis=0, b=data[..., None])
    log_replace_neginf(newlogC)
    newlogC -= logsumexp(newlogC, axis=(0, 1), keepdims=True)

    return newlogW, newlogC


################# Initializer ####################
class MDPD_initializer():
    @classmethod
    def init_random_uniform(cls, dim, ncomp, nvocab):
        logW = np.ones(ncomp) * (- np.log(ncomp))
        logC = np.log(np.random.uniform(low=0.2, high = 0.8, size = (dim, nvocab, ncomp)))
        logC = logC - logsumexp(logC, axis=1, keepdims=True)
        return logW, logC


class Crowdsourcing_initializer(MDPD_initializer):
    @classmethod
    def init_mv(cls, data, features, rm_last=False):  # majority vote
        data_selected = data[:, features, :]
        if rm_last:
            votes = np.sum(data_selected[..., :-1], axis=1, dtype=np.float)
        else:
            votes = data_selected.sum(axis=1, dtype=np.float)  # n-r
        log_votes = np.log(votes)
        # replace -inf with NINF
        log_replace_neginf(log_votes)
        log_votes_sum = logsumexp(log_votes, axis=1, keepdims=True)
        # normalize log_post
        log_post = log_votes - log_votes_sum
        return mstep(log_post, data)

    @classmethod
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


################ Feature Selection #####################
# MDPD feature selection
class Feature_Selection():
    # TODO to be deprecated
    # @classmethod
    # def MI_feature_selection(cls, data, topN):
    #     ranking, sigma = cls.MI_feature_ranking(data)
    #     return ranking[:topN], sigma[:topN]

    @classmethod
    def MI_feature_ranking(cls, data, lock=None):
        """
        Calculate sum_{j} sum_{x_i, x_j} P(x_i, x_j) ln(p(x_i, x_j) / p(x_i)p(x_j)) and sort
        :param data:
        :param lock:
        :return:
        """
        score = cls.MI_score(data, rm_diag=True, lock=lock)
        sigma = score.sum(axis=1)
        ranking = np.argsort(sigma, axis=None)[::-1]
        return ranking, sigma[ranking]

    @classmethod
    def pmi(cls, data, lock=None):
        """
        calculate P(x_i, x_j) ln(p(x_i, x_j) / p(x_i)p(x_j))
        :return: d - r - d - r
        """
        nsample, dim, nvocab = data.shape
        log_post = np.zeros((nsample, 1))
        pmi = cls.pmi_conditional(data, log_post, lock=lock)
        return pmi[..., 0]


    @classmethod
    def pmi_conditional(cls, data, log_post, lock=None, sample_log_weights=None):
        """
        calculate P(x_i, x_j|y=k) ln(p(x_i, x_j|y=k) / p(x_i|y=k)p(x_j|y=k))
        :param data:
        :param log_post:
        :return: d - r - d - r - c
        """
        nsample, dim, nvocab = data.shape
        ncomp = log_post.shape[1]
        newlogW, newlogC = mstep(log_post, data)

        if sample_log_weights is None:
            sample_weights = np.ones(nsample, dtype=np.float) / nsample
        else:
            sample_weights = np.exp(sample_log_weights)

        post = np.exp(log_post)
        data_transform = data[:, :, :, None] * np.sqrt(post)[:, None, None, :] * np.sqrt(sample_weights)[:, None, None, None]
        cache = []
        for k in range(ncomp):
            second = np.tensordot(data_transform[:, :, :, k], data_transform[:, :, :, k], axes=(0, 0))
            second = second / np.exp(newlogW[k])
            #
            if np.any(lock):
                mask = (lock[..., None, None] + lock[None, None, ...]) == 0
                # scaled log_second
                second += PESP
                second_masked = second * mask
                second_masked /= np.sum(second_masked, axis=(1, 3), keepdims=True)
                log_second_scaled = np.log(second_masked)
                log_replace_neginf(log_second_scaled)
                # scaled log_first P(x_i|y, x_i x_k \neq missing label)
                log_first_scaled = logsumexp(log_second_scaled, axis=3)
                log_first_scaled = log_first_scaled[..., None] + np.moveaxis(log_first_scaled, (0,1,2), (1,2,0))[:, None,...]
                pmi = second_masked * (log_second_scaled - log_first_scaled)
            else:
                log_first = np.add.outer(newlogC[:, :, k], newlogC[:, :, k])
                log_second = np.log(second)
                pmi = second * (log_second - log_first)
                pmi[second == 0] = 0
            # pmi[log_first == 0] = 0
            cache.append(pmi[..., None])
        return np.concatenate(cache, axis=4)


    # @classmethod
    # def pmi_conditional(cls, data, log_post, lock=None):
    #     """
    #     calculate P(x_i, x_j|y=k) ln(p(x_i, x_j|y=k) / p(x_i|y=k)p(x_j|y=k))
    #     :param data:
    #     :param log_post:
    #     :return: d - r - d - r - c
    #     """
    #     nsample, dim, nvocab = data.shape
    #     ncomp = log_post.shape[1]
    #     newlogW, newlogC = mstep(log_post, data)
    #     cache = []
    #     b = data[..., None, None] * data[:, None, None, ...]
    #     mask = (lock[..., None, None] + lock[None, None, ...]) == 0 if np.any(lock) else None
    #     for k in range(ncomp):
    #         if np.any(lock):
    #             log_second = logsumexp(log_post[:, None, None, None, None, k],
    #                                    axis=0, b=b) - np.log(nsample)
    #             normalizer = logsumexp(log_second, axis=(1, 3), b=mask, keepdims=True)
    #             # a mask show that x_i, x_j has at least one missing value at a time
    #             mask_miss = np.broadcast_to(np.isinf(normalizer), mask.shape)
    #             log_second_masked = log_second - normalizer
    #             joint_mask = np.logical_and(mask, mask_miss)
    #             log_second_masked[joint_mask] = np.broadcast_to(-np.log(np.sum(mask, axis=(1, 3), keepdims=True)) , mask.shape)[joint_mask]
    #             log_second_masked[np.logical_not(mask)] = NINF
    #             log_replace_neginf(log_second_masked)
    #             log_first_masked = logsumexp(log_second_masked, axis=3, b=mask)
    #             log_first_masked = log_first_masked[..., None] + np.moveaxis(log_first_masked, (0, 1, 2), (1, 2, 0))[:,
    #                                                              None, ...]
    #             pmi = np.exp(log_second_masked) * (log_second_masked - log_first_masked)
    #             pmi[np.logical_not(mask)] = 0
    #         else:
    #             log_first = newlogC[..., None, None, k] + newlogC[None, None, ..., k]
    #             log_second = logsumexp(log_post[:, None, None, None, None, k],
    #                                    axis=0, b=b) - np.log(nsample) - newlogW[k]
    #             pmi = np.exp(log_second) * (log_second - log_first)
    #             pmi[np.isinf(log_second)] = 0
    #         cache.append(pmi[..., None])
    #     return np.concatenate(cache, axis=4)


    @classmethod
    def MI_score(cls, data, rm_diag=False, lock=None):
        """
        Calculate sum_{x_i, x_j} P(x_i, x_j) ln(p(x_i, x_j) / p(x_i)p(x_j))
        :return:
        """
        pmi = cls.pmi(data, lock=lock)
        score = pmi.sum(axis=(1, 3))
        #
        # pmi = cls.pmi(data)
        # if np.any(lock):
        #     mask = (lock[..., None, None] + lock[None, None, ...]) == 0
        #     score = np.sum(pmi * mask, axis=(1, 3))
        # else:
        #     score = pmi.sum(axis=(1, 3))
        if rm_diag:
            np.fill_diagonal(score, 0)
        return score

    @classmethod
    def MI_score_conditional(cls, data, log_post, rm_diag=False, lock=None):
        """
        Calculate \sum_{x_i, x_j}P(x_i, x_j|y=k) ln(p(x_i, x_j|y=k) / p(x_i|y=k)p(x_j|y=k)).
        :param data:
        :param log_post: n - c
        :param rm_diag:
        :param lock: d - r
        :return: d - d - c, c
        """
        ncomp = log_post.shape[1]
        pmi = cls.pmi_conditional(data, log_post, lock=lock)
        newlogW, _ = mstep(log_post, data)
        score = np.sum(pmi, axis=(1, 3))
        # if np.any(lock):
        #     mask = (lock[..., None, None] + lock[None, None, ...]) == 0
        #     score = np.sum(pmi * mask[..., None], axis=(1, 3))
        # else:
        #     score = np.sum(pmi, axis=(1, 3))
        if rm_diag:
            for k in xrange(ncomp):
                np.fill_diagonal(score[..., k], 0)
        return score, np.exp(newlogW)


################# Utilities ####################


def log_replace_neginf(array):
    array[np.isneginf(array)] = NINF


def logsumexp(a, axis=None, b=None, keepdims=False):
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)

    out += a_max

    return out


# TODO: to be deprecated
# def mylog(input):
#     """
#
#     :param input:
#     :return:
#     """
#     if isinstance(input, int):
#         return np.log(input) if input != 0 else np.log(sys.float_info.min)
#     else:
#         if isinstance(input, list):
#             input = np.asarray(input)
#         _shape = input.shape
#         input = input.flatten()
#         _len = np.size(input)
#         output = np.zeros(_len)
#         for i in range(_len):
#             foo = input[i]
#             output[i] = np.log(foo) if foo != 0 else np.log(sys.float_info.min)
#         return output.reshape(_shape)

# def comp_duplicate(logW, logC, lock, comp):
#     newlogW = np.copy(logW)
#     newlogW = np.append(newlogW, newlogW[comp] - np.log(2))
#     newlogW[comp] -= np.log(2)
#
#     newlogC = np.copy(logC)
#     newlogC = np.append(newlogC, newlogC[:, :, comp][:, :, None], axis=2)
#
#     lock = np.append(lock, lock[:, :, comp][:, :, None], axis=2)
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
#             foo += logW[:, None]
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
