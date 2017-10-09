"""
The core module of the package.

Basic MDPD class is defined in this module which is composed of basic functions. Stage-wise EM is defined as a subclass of MDPD.

"""
from __future__ import division

import cPickle
import copy
import itertools
import logging
import numpy as np
from copy import deepcopy
from scipy.misc import logsumexp

import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(10)


class MDPD_basic(object):
    def __init__(self):
        '''constructor'''
        self.dim = None  # dimension
        # self.trace = None   # trace used by merge method
        # self.nsample = None  # number of sample
        self.ncomp = None  # number of components (or the target number of components)
        self.nvocab = None  # size of discrete alphabet
        # self.label = None
        self._logW = None  # log mixing weights
        self._logC = None  # log confusion matrix

    @property
    def logW(self):
        return self._logW

    @logW.setter
    def logW(self, val):
        self._logW = deepcopy(val)

    @property
    def logC(self):
        return self._logC

    @logC.setter
    def logC(self, val):
        self._logC = deepcopy(val)


    def get_config(self, dim=None, nsample=None, ncomp=None, nvocab=None):
        """Get model configuration"""
        self.dim = int(dim)
        self.nsample = int(nsample)
        self.ncomp = int(ncomp)
        self.nvocab = int(nvocab)


    def log_likelihood(self, data):
        foo = utils.log_joint_prob_fast(data, self._logW, self._logC)
        lsum = logsumexp(foo, axis=1)
        return lsum.mean()

    def log_posterior(self, data):
        foo = utils.log_joint_prob_fast(data, self._logW, self._logC)
        lsum = utils.logsumexp(foo, axis=0)
        return foo - lsum[np.newaxis, :]

    def predict(self, data):
        logpost = self.log_posterior(data=data)
        return np.argmax(logpost, axis=0)

    def score(self, data, label):
        pred = self.predict(data)
        if label.shape != pred.shape:
            raise ValueError('The shape of label is {}. However, {} is expected.'.format(label.shape, pred.shape))
        else:
            acc = sum(pred - label == 0) / label.size
            print 'Accuracy: {0:.2%}'.format(acc)

    def reorder(self, order):
        """Perform label swap according to 'order'."""
        if len(order) != len(self._logW):
            raise ValueError('The size of order is {}. However, {} is expected.'.format(len(order), len(self._logW)))
        self._logW = self._logW[order]
        self._logC = self._logC[:, :, order]

    def _get_tbs(self):
        """helper for save"""
        tbs = {}  # to be saved
        tbs['m'] = self.dim
        tbs['n'] = self.nsample
        tbs['c'] = self.ncomp
        tbs['r'] = self.nvocab
        tbs['z'] = self.data
        tbs['label'] = self.label
        tbs['W'] = self._logW
        tbs['C'] = self._logC
        tbs['Wgen'] = self.Wgen
        tbs['Cgen'] = self.Cgen
        tbs['trace'] = self.trace
        return tbs

    def save(self, filename):
        """
        This method is to save the model by using pickle (cpickle).

        Attributes to be saved will be collected into a single dictionary and then save to the file named 'filename'.
        Attributes to be saved are: m, n, c, r, z, label, W, C, Wgen, Cgen
        :param filename:
        :return:
        """
        if filename[-2:] != '.p':
            filename += '.p'

        tbs = self._get_tbs()

        with open(filename, 'wb') as f:
            cPickle.dump(tbs, f)

    def _load(self, tbl):
        """Helper for load"""
        self.dim = tbl['m']
        self.nsample = tbl['n']
        self.ncomp = tbl['c']
        self.nvocab = tbl['r']
        self.data = tbl['z']
        self.label = tbl['label']
        self._logW = tbl['W']
        self._logC = tbl['C']
        self.Wgen = tbl['Wgen']
        self.Cgen = tbl['Cgen']
        self.trace = tbl['trace']

    def load(self, filename):
        """
        This methods is to load the model from 'filename'
        :param filename:
        :return:
        """
        if filename[-2:] != '.p':
            filename += '.p'
        with open(filename, 'rb') as f:
            tbl = cPickle.load(f)
            self._load(tbl)

            # def merge(self, cord_list):
            #     _num = len(cord_list)
            #     newm = self.dim - _num + 1
            #     newr = 1
            #     newtrace = []
            #     foo = []    # for z
            #     bar = [1]
            #     for i in cord_list:
            #         newr *= self.nvocab[i]
            #         newtrace += self.trace[i]
            #         # binary coding to numerics
            #         foo.append(np.dot(np.asarray(range(self.nvocab[i])), self.data[i]))
            #         bar.append(bar[-1] * self.nvocab[i])
            #     bar.pop()
            #     bar = np.asarray(bar).reshape((_num, 1))
            #     foo = np.asarray(foo)   # numeric responses of each in cord_list
            #     numeric_res = np.sum(foo * bar, axis=0) # numeric responses to each sample for the merged coordinate
            #     # convert to binary encoding
            #     foo = numeric_res * self.nsample + np.asarray(range(self.nsample))
            #     newz = np.zeros(newr * self.nsample)
            #     newz[foo] = 1
            #     newz = newz.reshape((newr, self.nsample))
            #     # newC and newCgen
            #     newC = np.zeros((newr, self.ncomp))
            #     if self.Cgen:
            #         newCgen = np.zeros((newr, self.ncomp))
            #     for k in range(self.ncomp):
            #         cord_list_copy = copy.copy(cord_list)
            #         id = cord_list_copy.pop()
            #         foo1 = self.logC[id][:, k].flatten()
            #         if self.Cgen:
            #             foo2 = self.Cgen[id][:, k].flatten()
            #         for count in range(1, _num):
            #             id = cord_list_copy.pop()
            #             foo1 = np.outer(foo1, self.logC[id][:, k]).flatten()
            #             if self.Cgen:
            #                 foo2 = np.outer(foo2, self.logC[id][:, k]).flatten()
            #         newC[:, k] = foo1
            #         if self.Cgen:
            #             newCgen[:, k] = foo2
            #     # clean up model
            #     self.dim = newm
            #     self.nvocab.append(newr)
            #     self.nvocab[:] = [item for i, item in enumerate(self.nvocab) if i not in cord_list]
            #     self.data.append(newz)
            #     self.data[:] = [item for i, item in enumerate(self.data) if i not in cord_list]
            #     self.logC.append(newC)
            #     self.logC = [item for i, item in enumerate(self.logC) if i not in cord_list]
            #     if self.Cgen:
            #         self.Cgen.append(newCgen)
            #         self.Cgen = [item for i, item in enumerate(self.Cgen) if i not in cord_list]
            #     self.trace.append(newtrace)
            #     self.trace = [item for i, item in enumerate(self.trace) if i not in cord_list]


class MDPD(MDPD_basic, object):
    def __init__(self):
        MDPD_basic.__init__(self)
        # nsample, dim, nvocab = data.shape
        # self.nsample = nsample
        # self.dim = dim
        # self.nvocab = nvocab
        # self.ncomp = ncomp
        ## core attributes
        self.feature_set = []
        # self.MIres = None
        ## temporary attributes
        self.lock = None

    def init_topNfeatures(self, data, topN, remove_last=False):
        self.reset(data)
        MIcomplete = self.get_MIcomplete(data)
        if remove_last:
            MIcomplete = np.delete(MIcomplete, -1, axis=1)
            MIcomplete = np.delete(MIcomplete, -1, axis=3)
        foo = MIcomplete.sum(axis=(1, 3))
        np.fill_diagonal(foo, 0)
        score = foo.sum(axis=1)
        bar = np.argsort(score, axis=None)[::-1]
        return bar[:topN], score[bar]

    # def init_beta(self, data, topN, ncomp):
    #     infoset = self.init_topNfeatures(data, topN)
    #     data_info = data[:, infoset, :]
    #     model_spec = MDPD.MDPD(data_info, ncomp)
    #     model_spec.train(data_info, method="spectral", niter=1)
    #     post = model_spec.logposterior(subset=range(model_spec.dim))
    #     newlogW, newlogC = self.Mstep(post, data)
    #     self.logW = newlogW
    #     self.logC = self.logC
    #     self.__init__(data, ncomp=ncomp)


    def train(self, data, ncomp, method="majority", niter=30, diffthre=0.001, init=True,
              display=True, features=-1):
        nsample, dim, nvocab = data.shape
        if init:
            self.dim, self.nvocab = dim, nvocab
            self.ncomp = ncomp
        if features == -1:
            self.feature_set = range(dim)
        else:
            self.feature_set = features

        logger.info(
            "Training an MDPD with dimension %i, sample size %i, vocab size %i and the target number of components %i",
            self.dim, nsample, self.nvocab, self.ncomp)
        # choosing fitting methods
        if method == "majority":
            self.logW, self.logC = utils.init_mv(data, self.feature_set)
        elif method == "plain":
            self.logW, self.logC = utils.init_random(self.dim, self.ncomp, self.nvocab)
        elif method == "spectral":
            self.logW, self.logC = utils.init_spectral(data, self.ncomp)
        # statistics
        ll = self.loglikelihood(data)
        for count in xrange(niter):
            self.EM(data)
            ll = self.log_likelihood(data)
            logger.info("iteration %d; log-likelihood %f;", count, ll)

        # count = 0
        #
        # while True:
        #     count += 1
        #     self.EM(data, range(self.dim))
        #     ll = self.loglikelihood(data)
        #     logger.info("iteration %d; log-likelihood %f;", count, ll)
        #     if count >= niter:
        #         break

    def reset(self, data):
        self.feature_set = []
        self.logC = np.log(data.mean(axis=0).squeeze()[:, :, np.newaxis])
        self.logC[np.isinf(self.logC)] = -100
        self.logC -= logsumexp(self.logC, axis=1)[:, np.newaxis, :]
        self.lock = np.ones(self.logC.shape)
        # self.C = [data[i].mean(axis=1)[:, np.newaxis] for i in range(self.dim)]
        self.logW = np.asarray([0])
        self.ncomp = 1
        pass

    # def init_random(self):  # random init
    #     self.logW = np.ones(self.ncomp) / self.ncomp
    #     self.logC = np.random.random_sample((self.dim, self.nvocab, self.ncomp))
    #     self.logC /= self.logC.sum(axis=1)[:, np.newaxis, :]
    #     if self.lock is None:
    #         self.lock = np.zeros(self.logC.shape)
    #
    # def init_mv(self, data):  # majority vote
    #     """
    #
    #     :return:
    #     """
    #     # make sure all workers has same output size
    #     foo = data.sum(axis=1).astype(np.float)
    #     logpost = np.log(foo / foo.sum(axis=1)[:, np.newaxis]).T
    #     logpost[np.isinf(logpost)] = -100
    #     # foo = foo.argmax(axis=1)
    #     # logpost = np.ones([data.shape[2] ,data.shape[0]]) * -100
    #     # logpost[foo, range(data.shape[0])] = 0
    #     # soft
    #     logW, logC = self.mstep(logpost, data)
    #     self.logW = logW
    #     self.logC = logC
    #     if self.lock is None:
    #         self.lock = np.ones(self.logC.shape)
    #
    # def init_spec(self, data):
    #     """
    #     Use spectral methods to initialize EM. (Zhang. and Jordan. 2014)
    #     :param data:
    #     :return:
    #     """
    #     # divide workers into 3 partitions
    #     num_worker_partition = int(self.dim / 3)
    #     # np.random.seed(seed=10)
    #     foo = np.random.permutation(self.dim)
    #
    #     partition = [None] * 3
    #     partition[0] = foo[:num_worker_partition]
    #     partition[1] = foo[num_worker_partition:2 * num_worker_partition]
    #     partition[2] = foo[2 * num_worker_partition:]
    #
    #     # calculate average response to each sample for each group
    #     train_g = [None] * 3
    #     for g in range(3):
    #         foo = data[:, partition[g], :]
    #         train_g[g] = np.mean(foo, axis=1).T
    #
    #     #
    #     perm = [[1, 2, 0], [2, 0, 1], [0, 1, 2]]
    #     #
    #     W = []
    #     Cg = []
    #     for g in range(3):
    #         a = perm[g][0]
    #         b = perm[g][1]
    #         c = perm[g][2]
    #         m2, m3 = get_tensor(train_g, a, b, c)
    #         foo, bar = tensorpower(m2, m3, self.ncomp)
    #         W.append(foo)
    #         Cg.append(bar)
    #     W = np.mean(np.asarray(W), axis=0)
    #     # normalize W
    #     W /= sum(W)
    #     # use Cg to recover C for each worker
    #     C = np.zeros((self.dim, self.nvocab, self.ncomp))
    #     for i in range(self.dim):
    #         foo = np.zeros((self.ncomp, self.ncomp))
    #         for g in range(3):
    #             if i not in partition[g]:
    #                 foo += get_Ci(train_g, W, Cg, g, data, i)
    #         foo = foo / 2
    #         C[i, :, :] = foo
    #         # C.append(foo)
    #     self.logC = np.log(C)
    #     self.logW = np.log(W)

    # fine tune
    def refine(self, data, infoset=None, niter=20):
        logger.info('Fine tune the model with ')
        infoset = infoset or range(self.dim)
        for count in range(niter):
            self.EM(data, infoset)

    # # split component
    # def split(self, data, cord1, cord2, comp):
    #     # get Hessian (Hessian of marginal loglikelihood)
    #     H = get_Hessian2(data, self.logW, self.logC, self.lock, self.feature_set, cord1, cord2, comp)
    #     # solve constrained singular value problem
    #     U, Sig, V = np.linalg.svd(H)
    #     V = V.T
    #     m, n = U.shape
    #     allone = np.ones(m) * 1 / np.sqrt(m)
    #     for i in xrange(n):
    #         U[:, i] -= np.dot(U[:, i], allone) * allone
    #     m, n = V.shape
    #     allone = np.ones(m) * 1 / np.sqrt(m)
    #     for i in xrange(n):
    #         V[:, i] -= np.dot(V[:, i], allone) * allone
    #
    #     H_star = U.dot(np.diag(Sig)).dot(V.T)
    #     U, Sig, V = np.linalg.svd(H_star)
    #     V = V.T
    #
    #     #
    #     dp1 = U[:, 0]
    #     dp2 = V[:, 0]
    #
    #     # split component k into two
    #     self.logW, self.logC, self.lock, self.ncomp = comp_duplicate(self.logW, self.logC, self.lock, comp)
    #     ## break symmetry
    #     stepsize = .05
    #
    #     def add(logC, lock, cord, comp, dp):
    #         prob = np.exp(logC[cord, lock[cord, :, comp] == 1, comp]) + dp
    #         prob[prob < 0] = 0.001
    #         prob[prob > 1] = 1
    #         logC[cord, lock[cord, :, comp] == 1, comp] = np.log(prob)
    #
    #     add(self.logC, self.lock, cord1, comp, dp1 * stepsize)
    #     add(self.logC, self.lock, cord1, -1, -dp1 * stepsize)
    #     add(self.logC, self.lock, cord2, comp, dp2 * stepsize)
    #     add(self.logC, self.lock, cord2, -1, -dp2 * stepsize)

    ## EM step
    def EM(self, data):
        #
        lock = self.lock
        # E-step
        post = self.log_posterior(data)
        # M-step (full M-step)
        newlogW, newlogC = utils.mstep(post, data)
        if np.any(lock == 0):
            newlogC[lock == 0] = -100
            self.logC[lock == 1] = -100
            newlogC = newlogC - logsumexp(newlogC, axis=1)[:, np.newaxis, :] \
                      + np.log(1 - np.exp(logsumexp(self.logC, axis=1)))[:, np.newaxis, :]
            newlogC[lock == 0] = self.logC[lock == 0]
        self.logW = newlogW
        self.logC = newlogC

    # def get_MIcomplete(self, data):
    #     return get_MIcomplete(data)

    ## get mutual inforamtion conditional on component (m-m-n-c)
    def get_MIres(self, data, rm_diag=False, wPMI=False):
        data_select = data[:, self.feature_set, :]
        logpost = self.log_posterior(data)
        return utils.MIres_fast(data_select, logpost, rm_diag=rm_diag, wPMI=wPMI)

    # ## get MI
    # def get_MI(self, data, rm_diag=False, subset=None):
    #     return get_MI(data, self.logW, self.logC, subset or self.feature_set, rm_diag=rm_diag)

    def log_posterior(self, data):
        data_selected = data[:, self.feature_set, :]
        return MDPD_basic.log_posterior(self, data_selected)

    def predict(self, data):
        data_selected = data[:, self.feature_set, :]
        return MDPD_basic.predict(self, data_selected)

    def score(self, data, label):
        data_selected = data[:, self.feature_set, :]
        return MDPD_basic.score(self, data_selected, label)


    # ## predict (overwrite)
    # def predict(self, data, label, subset=None):
    #     """Handle Ties"""
    #     logpost = self.log_posterior(data, subset=subset or self.feature_set)
    #     # new
    #     rank = np.argsort(logpost, axis=0)[::-1, :]
    #     if data.shape[0] != len(label):
    #         return rank[0, :]
    #     else:
    #         err = 0
    #         tie = 0
    #         for i in xrange(data.shape[0]):
    #             idx = 0
    #             while idx < rank.shape[0] - 1 and logpost[rank[idx + 1, i], i] == logpost[rank[0, i], i]:
    #                 idx += 1
    #             if label[i] not in rank[:idx + 1, i]:
    #                 err += 1
    #             else:
    #                 err += idx / (idx + 1)
    #                 if idx:
    #                     tie += 1
    #         print 'error rate is ' + str(err / len(label))
    #         print str(tie) + ' ties'
    #         return rank[0, :], err / len(label)
    #         # old
    #         # foo = np.argmax(logpost, axis=0)
    #         # if data.shape[0] != len(label):
    #         #     return foo
    #         # else:
    #         #     err = sum(foo-label!=0)
    #         #     print 'error rate is ' + str(err/len(label))
    #         #     return foo, err / len(label)

    # ## Swap
    # def swap(self, idx1, idx2):
    #     foo = self.logW[idx1]
    #     self.logW[idx1] = self.logW[idx2]
    #     foo = np.copy(self.logC[:, :, idx1])
    #     self.logC[:, :, idx1] = self.logC[:, :, idx2]
    #     self.logC[:, :, idx2] = foo

    def align(self, data, label, subset):
        pred, err = self.predict(data, label, subset=subset)
        logW = self.logW
        logC = self.logC
        best_order = None
        best_score = 1
        for order in itertools.permutations(range(self.ncomp)):
            order = list(order)
            self.reorder(order)
            _, err = self.predict(data, label, subset=subset)
            if err < best_score:
                best_order = order
                best_score = err
            # restore
            self.logW = logW
            self.logC = logC
        # print order
        self.reorder(best_order)

    # def add_infoset(self, input):
    #     return add_infoset(self._feature_set, input)

    def save(self, filename):
        """override save"""
        if filename[-2:] != '.p':
            filename += '.p'
        tbs = self._get_tbs()
        tbs['activeset'] = self.feature_set
        with open(filename, 'wb') as f:
            cPickle.dump(tbs, f)

    def load(self, filename):
        """override load"""
        if filename[-2:] != '.p':
            filename += '.p'
        with open(filename, 'rb') as f:
            tbl = cPickle.load(f)
            self._load(tbl)
            self.feature_set = tbl['activeset']

    # def merge(self, cord_list):
    #     """
    #
    #     :param cord_list:
    #     :return:
    #     """
    #     _num = len(cord_list)
    #     newm = self.dim - _num + 1
    #     idx = [item for i, item in enumerate(range(self.dim)) if i not in cord_list]
    #     newactiveset = []
    #     for i in self._feature_set:
    #         if i not in cord_list:
    #             newactiveset.append(idx.index(i))
    #         else:
    #             if newm - 1 not in newactiveset:
    #                 newactiveset.append(newm - 1)
    #     self._feature_set = newactiveset
    #     MDPD_basic.merge(self, cord_list)
