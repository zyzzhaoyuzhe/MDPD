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

NINF = np.finfo('f').min

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
        log_marginal_x = logsumexp(foo, axis=1)
        return log_marginal_x.mean()

    def log_posterior(self, data):
        log_joint = utils.log_joint_prob_fast(data, self._logW, self._logC)
        normalizer = logsumexp(log_joint, axis=1, keepdims=True)
        return log_joint - normalizer

    def predict(self, data):
        log_post = self.log_posterior(data)
        return np.argmax(log_post, axis=1)

    def accuracy(self, data, label):
        pred = self.predict(data)
        if label.shape != pred.shape:
            raise ValueError('The shape of label is {}. However, {} is expected.'.format(label.shape, pred.shape))
        else:
            acc = sum(pred - label == 0) / label.size
            logger.info('ACCURACY: {0:.2%}'.format(acc))
            return acc

    def MI_residue(self, data):
        log_post = self.log_posterior(data)
        score, weights = utils.Feature_Selection.MI_score_conditional(data, log_post, rm_diag=True)
        res = np.sum(score.sum(axis=1) * weights) / (self.dim * (self.dim - 1))
        logger.info('The mutual information residue is {}'.format(res))

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


class MDPD_standard(MDPD_basic):
    """MDPD model with batch EM + Feature Selection"""
    def __init__(self):
        super(MDPD_standard, self).__init__()
        self.features = []
        self.lock = None

    def MI_residue(self, data):
        log_post = self.log_posterior(data)
        score, weights = utils.Feature_Selection.MI_score_conditional(data, log_post, rm_diag=True, lock=self.lock)
        res = np.sum(score.sum(axis=1) * weights[np.newaxis, :]) / (self.dim * (self.dim - 1))
        logger.info('The mutual information residue (include all features) is {}'.format(res))

        features = np.array(self.features)
        score_select = score[features[:, np.newaxis], features, :]
        res_select = np.sum(score_select.sum(axis=1) * weights[np.newaxis, :]) / (len(features) * (len(features) - 1))
        logger.info('The mutual information residue (within selected features) is {}'.format(res_select))

    def _apply_lock(self, data):
        "Apply the lock to worker i for the component k, so that i is not discriminant at k."
        # data = data[:, self.features, :]
        lock = self.lock
        if np.any(lock):
            newlogC = self.logC.copy()
            lock_broadcast = np.broadcast_to(lock[..., np.newaxis], newlogC.shape)
            newlogC[lock_broadcast] = NINF
            log_margin_prob = np.log(data.sum(axis=0) / data.shape[0])
            utils.log_replace_neginf(log_margin_prob)
            log_margin_prob_sum = logsumexp(log_margin_prob, axis=1, keepdims=True, b=1 - lock)[..., np.newaxis]
            newlogC_sum = logsumexp(newlogC, axis=1, keepdims=True)
            newlogC = newlogC - newlogC_sum + log_margin_prob_sum
            newlogC[lock_broadcast] = np.broadcast_to(log_margin_prob[..., np.newaxis], newlogC.shape)[lock_broadcast]
            self.logC = newlogC

    def _model_init(self, data, init, init_label, init_para):
        nsample, dim, nvocab = data.shape
        if isinstance(init, basestring):
            if init == "majority":
                self.logW, self.logC = utils.Crowdsourcing_initializer.init_mv(data, self.features,
                                                                               rm_last=np.any(self.lock))
            elif init == "random":
                self.logW, self.logC = utils.Crowdsourcing_initializer.init_random_uniform(dim, self.ncomp, self.nvocab)
            elif init == "spectral":
                self.logW, self.logC = utils.init_spectral(data, self.ncomp)
            else:
                raise ValueError('init is not valid. It needs to one of "majority", "random", and "spectral"')
        elif isinstance(init_label, np.ndarray):
            self.logW, self.logC = utils.mstep(init_label, data[:, self.features, :])
        elif isinstance(init_para, (tuple,list)):
            self.logW, self.logC = init_para
        else:
            raise ValueError('No valid initialization.')

    def fit(self, data, ncomp,
            features=None, init=None,
            init_label=None, init_para=None,
            epoch=30, update_features_per_epoch=None,
            verbose=True, lock=None):
        """
        Fit the model to training data.
        :param data: numpy array, shape = (nsample, dim, nvocab)
        :param ncomp: int, number of components
        :param features: list of ints, selected features
        :param init:
        :param init_label:
        :param init_para:
        :param epoch:
        :param verbose:
        :return:
        """
        nsample, dim, nvocab = data.shape
        self.dim, self.nvocab, self.ncomp = dim, nvocab, ncomp
        # initiate features
        if features is None:
            self.features = range(dim)
        elif isinstance(features, (list, np.ndarray)):
            self.features = features
        elif isinstance(features, int):
            cand, _ = utils.Feature_Selection.MI_feature_ranking(data, lock=lock)
            self.features = cand[:features]
        else:
            raise ValueError('invalid input type for <features>')
        # self.features = features if features is not None else range(dim)
        self.lock = np.array(lock, dtype=np.bool) if lock is not None else np.zeros((dim, nvocab), dtype=np.bool)
        logger.info(
            "Training an MDPD with dimension %i, %i features, sample size %i, vocab size %i and the target number of components %i",
            self.dim, len(self.features), nsample, self.nvocab, self.ncomp)
        self._model_init(data, init, init_label, init_para)
        self._apply_lock(data)
        # statistics
        self._em_wrapper(data, epoch, update_features_per_epoch, verbose=verbose)

    def _em(self, data):
        """One step EM iteration"""
        # E-step
        log_post = self.log_posterior(data)
        # M-step (full M-step)
        self.logW, self.logC = utils.mstep(log_post, data)
        self._apply_lock(data)

    def _em_wrapper(self, data, epoch, update_features_per_epoch, verbose=False):
        for ep in xrange(epoch):
            if ep > 0 and update_features_per_epoch is not None and ep % update_features_per_epoch == 0:
                self._update_features(data)
            self._em(data)
            if verbose:
                self._verbose_printer(ep, data)

    def _update_features(self, data):
        """update features according to conditional information residue"""
        log_post = self.log_posterior(data)
        score, weights = utils.Feature_Selection.MI_score_conditional(data, log_post, rm_diag=True, lock=self.lock)
        sigma = np.sum(score.sum(axis=1) * weights[np.newaxis, :], axis=1)
        cand = np.argsort(sigma)[::-1]
        self.features = cand[:len(self.features)]

    def _verbose_printer(self, ep, data):
        logger.info("iteration %d; log-likelihood (feature selection) %f; log_likelihood %f", ep,
                    self.log_likelihood(data), super(MDPD_standard, self).log_likelihood(data))

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

    #
    def change_features(self, data, features=None):
        """Change the feature set"""
        features = features or range(self.dim)
        self._assert_features(features)
        self.features = features

    # fine tune
    def refine(self, data, features=None, niter=20, verbose=False):
        """
        Fine tune the model
        :param data:
        :param features:
        :param niter:
        :return:
        """
        new_features = features or range(self.dim)
        self._assert_features(new_features)
        if set(new_features) != set(self.features):
            logger.info('Fine tune the model with the feature set {}'.format(new_features))
            self.change_features(data, features=new_features)
            self.features = new_features
        else:
            logger.info('Fine tune the model.')
        self._em_wrapper(data, niter, verbose=verbose)

    @staticmethod
    def _assert_features(features):
        assert not isinstance(features, basestring)

    def log_likelihood(self, data):
        data_selected = data[:, self.features, :]
        log_joint = utils.log_joint_prob_fast(data_selected, self.logW, self.logC[self.features, ...])
        log_marginal_x = logsumexp(log_joint, axis=1)
        return log_marginal_x.mean()

    def log_posterior(self, data):
        data_selected = data[:, self.features, :]
        log_joint = utils.log_joint_prob_fast(data_selected, self.logW, self.logC[self.features, ...])
        normalizer = logsumexp(log_joint, axis=1, keepdims=True)
        return log_joint - normalizer

    # def predict(self, data):
    #     data_selected = data[:, self.features, :]
    #     return MDPD_basic.predict(self, data_selected)

    # def accuracy(self, data, label):
    #     data_selected = data[:, self.features, :]
    #     return MDPD_basic.accuracy(self, data_selected, label)

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
            acc = self.accuracy(data_selected, label)
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


class MDPD2(MDPD_basic):
    def __init__(self):
        super(MDPD2, self).__init__()
        self.features_comp = None   # feature sets per mixture component

    def fit(self, data, ncomp, Ntop, init='random', batch=100, update_feature_per_batchs=50, epoch=50, verbose=True):
        "Fit the model to data use independent feature sets for each components. The algorithm will update the feature sets every a number of batches."
        ## update the instance parameters
        nsample, dim, nvocab = data.shape
        self.dim, self.nvocab, self.ncomp = dim, nvocab, ncomp
        self.Ntop = Ntop
        logger.info(
            "Training an MDPD with dimension %i, %i features, sample size %i, vocab size %i and the target number of components %i",
            dim, Ntop, nsample, self.nvocab, self.ncomp)
        ## initialize
        if init == 'random':
            self.logW, self.logC = utils.MDPD_initializer.init_random_uniform(self.dim, self.ncomp, self.nvocab)

        rank, sigma = utils.Feature_Selection.MI_feature_ranking(data[:1000, ...])
        self.features_comp = [rank[:Ntop] for _ in xrange(ncomp)]

        ## learning
        nbatch = int(nsample / batch)
        for ep in xrange(epoch):
            # random permutate the data
            data = data[np.random.permutation(data.shape[0]), ...]
            for t in xrange(nbatch):
                idx_batch = np.arange(t*batch, (t+1)*batch)
                if t % update_feature_per_batchs == 0 and ep + t > 0:
                    self._update_features_comp(data[idx_batch, ...])
                self._em(data[idx_batch, ...])

    def _update_features_comp(self, data_batch):
        log_post = self.log_posterior(data_batch)
        score, _ = utils.Feature_Selection.MI_score_conditional(data_batch, log_post, rm_diag=True)
        score = np.sum(score, axis=1)
        for k in xrange(self.ncomp):
            self.features_comp[k] = np.argsort(score[:, k])[::-1][:self.Ntop]

    def _em(self, data_batch):
        # E-step
        log_post = self.log_posterior(data_batch)
        # M-step
        self.logW, self.logC = utils.mstep(log_post, data_batch)

    def log_posterior(self, data_batch):
        log_joint_prob = []
        for k in xrange(self.ncomp):
            features = self.features_comp[k]
            foo = utils.log_joint_prob_slice(data_batch[:, features, :], self.logW[k], self.logC[features, :, k])
            log_joint_prob.append(foo[:, np.newaxis])
        log_joint_prob = np.concatenate(log_joint_prob, axis=1)
        normalizer = logsumexp(log_joint_prob, axis=1, keepdims=True)
        return log_joint_prob - normalizer  





