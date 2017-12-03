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
        self.ncomp = None  # number of components (or the target number of components)
        self.nvocab = None  # size of discrete alphabet
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
        lsum = utils.logsumexp(foo, axis=1)
        return foo - lsum[:, np.newaxis]

    def predict(self, data):
        logpost = MDPD_basic.log_posterior(self, data)
        return np.argmax(logpost, axis=1)

    def score(self, data, label):
        pred = MDPD_basic.predict(self, data)
        if label.shape != pred.shape:
            raise ValueError('The shape of label is {}. However, {} is expected.'.format(label.shape, pred.shape))
        else:
            acc = sum(pred - label == 0) / label.size
            logger.info('ACCURACY: {0:.2%}'.format(acc))
            return acc

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
        super(MDPD, self).__init__()
        self.features = []
        self.lock = None

    def fit(self, data, ncomp,
            features=None, init="majority",
            init_label=None, init_para=None,
            niter=30, verbose=True):
        """
        Fit the model to training data.
        :param data: numpy array, shape = (nsample, dim, nvocab)
        :param ncomp: int, number of components
        :param features: list of ints, selected features
        :param init:
        :param init_label:
        :param init_para:
        :param niter:
        :param verbose:
        :return:
        """
        nsample, dim, nvocab = data.shape
        logger.info(
            "Training an MDPD with dimension %i, sample size %i, vocab size %i and the target number of components %i",
            self.dim, nsample, self.nvocab, self.ncomp)
        if init:
            self.dim, self.nvocab = dim, nvocab
            self.ncomp = ncomp
        self.features = features if features is not None else range(dim)
        # choose initialization method
        if sum(map(bool, [init, init_label, init_para])) != 1:
            raise ValueError('Use one and only one of init, init_label, init_para.')
        if init == "majority":
            self.logW, self.logC = utils.init_mv(data, self.features)
        elif init == "random":
            self.logW, self.logC = utils.init_random(self.dim, self.ncomp, self.nvocab)
        elif init == "spectral":
            self.logW, self.logC = utils.init_spectral(data, self.ncomp)
        if init_label:
            self.logW, self.logC = utils.mstep(init_label, data[:, self.features, :])
        if init_para:
            self.logW, self.logC = init_para
        # statistics
        self._em_iterations(data, niter, verbose=verbose)


    def _em(self, data):
        """
        One step EM iteration
        :param data:
        :return:
        """
        data_selected = data[:, self.features, :]
        lock = self.lock
        # E-step
        logpost = self.log_posterior(data)
        # M-step (full M-step)
        newlogW, newlogC = utils.mstep(logpost, data_selected)
        if np.any(lock == 0):
            newlogC[lock == 0] = -100
            self.logC[lock == 1] = -100
            newlogC = newlogC - logsumexp(newlogC, axis=1)[:, np.newaxis, :] \
                      + np.log(1 - np.exp(logsumexp(self.logC, axis=1)))[:, np.newaxis, :]
            newlogC[lock == 0] = self.logC[lock == 0]
        self.logW = newlogW
        self.logC = newlogC

    def _em_iterations(self, data, niter, verbose=False):
        for count in xrange(niter):
            self._em(data)
            if verbose:
                logger.info("iteration %d; log-likelihood %f;", count, self.log_likelihood(data))
        # count = 0
        #
        # while True:
        #     count += 1
        #     self.EM(data, range(self.dim))
        #     ll = self.loglikelihood(data)
        #     logger.info("iteration %d; log-likelihood %f;", count, ll)
        #     if count >= niter:
        #         break

    # def reset(self, data):
    #     self.feature_set = []
    #     self.logC = np.log(data.mean(axis=0).squeeze()[:, :, np.newaxis])
    #     self.logC[np.isinf(self.logC)] = -100
    #     self.logC -= logsumexp(self.logC, axis=1)[:, np.newaxis, :]
    #     self.lock = np.ones(self.logC.shape)
    #     # self.C = [data[i].mean(axis=1)[:, np.newaxis] for i in range(self.dim)]
    #     self.logW = np.asarray([0])
    #     self.ncomp = 1
    #     pass



    # fine tune
    def refine(self, data, features=None, niter=20):
        """
        Fine tune the model
        :param data:
        :param features:
        :param niter:
        :return:
        """
        logger.info('Fine tune the model with ')
        foo, self.features = self.features, features or range(self.dim)
        self._em_iterations(data, niter, verbose=False)
        self.features = foo

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


    # def get_MIcomplete(self, data):
    #     return get_MIcomplete(data)

    ## get mutual inforamtion conditional on component (m-m-n-c)
    def get_MIres(self, data, rm_diag=False, wPMI=False):
        data_select = data[:, self.features, :]
        logpost = self.log_posterior(data)
        return utils.MIres_fast(data_select, logpost, rm_diag=rm_diag, weighted=wPMI)

    # ## get MI
    # def get_MI(self, data, rm_diag=False, subset=None):
    #     return get_MI(data, self.logW, self.logC, subset or self.feature_set, rm_diag=rm_diag)

    def log_likelihood(self, data):
        data_selected = data[:, self.features, :]
        return MDPD_basic.log_likelihood(self, data_selected)

    def log_posterior(self, data):
        data_selected = data[:, self.features, :]
        return MDPD_basic.log_posterior(self, data_selected)

    def predict(self, data):
        data_selected = data[:, self.features, :]
        return MDPD_basic.predict(self, data_selected)

    def score(self, data, label):
        data_selected = data[:, self.features, :]
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

    def align(self, data, label, features=None):
        """

        :param data:
        :param label:
        :param features:
        :return:
        """
        features = features or self.features
        data_selected = data[:, features, :]
        # pred = self.predict(data_selected)
        # acc = self.score(data_selected, label)
        oldlogW = self.logW
        oldlogC = self.logC
        best_order = None
        best_acc = 0
        for order in itertools.permutations(range(self.ncomp)):
            order = list(order)
            self.reorder(order)
            acc = self.score(data_selected, label)
            # _, err = self.predict(data, label)
            if acc > best_acc:
                best_order = order
                best_acc = acc
            # restore
            self.logW, self.logC = oldlogW, oldlogC
        # print order
        logger.info('Swap the components by {}.'.format(best_order))
        self.reorder(best_order)

    # def add_infoset(self, input):
    #     return add_infoset(self._feature_set, input)

    def save(self, filename):
        """override save"""
        if filename[-2:] != '.p':
            filename += '.p'
        tbs = self._get_tbs()
        tbs['features'] = self.features
        with open(filename, 'wb') as f:
            cPickle.dump(tbs, f)

    def load(self, filename):
        """override load"""
        if filename[-2:] != '.p':
            filename += '.p'
        with open(filename, 'rb') as f:
            tbl = cPickle.load(f)
            self._load(tbl)
            self.features = tbl['features']

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
