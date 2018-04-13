"""
The core module of the package.

Basic MDPD class is defined in this module which is composed of basic functions. Stage-wise EM is defined as a subclass of MDPD.

"""
from __future__ import division

import cPickle
import os
import copy
import itertools
import tempfile
from collections import defaultdict
import logging
import numpy as np
from copy import deepcopy
from utils import logsumexp

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

    def _tbs(self):
        """instance variables to be saved and loaded"""
        names = ['dim',
                 'ncomp',
                 'nvocab',
                 'logW',
                 'logC'
                 ]

        tbs = {}
        for name in names:
            tbs[name] = getattr(self, name)

        return tbs

    def save(self, filename):
        "This method is to save the model by using pickle (cpickle)."

        if filename[-2:] != '.p':
            filename += '.p'

        tbs = self._tbs()

        with open(filename, 'wb') as f:
            cPickle.dump(tbs, f)

    def load(self, filename):
        "This methods is to load the model from 'filename'"

        with open(filename, 'rb') as f:
            tbl = cPickle.load(f)
            self.load_dic(tbl)

    def load_dic(self, tbl):
        "Load the model from the dictionary tbl."
        for key, val in tbl.iteritems():
            setattr(self, key, val)


class MDPD_standard(MDPD_basic):
    """MDPD model with batch EM + Feature Selection"""
    def __init__(self, folder=None, name=None):
        super(MDPD_standard, self).__init__()
        self.features = []
        self.lock = None
        self._cache = defaultdict(list)
        self._folder = os.path.abspath('../' if folder is None else folder)
        self._cache['name'] = name
        self._sample_log_weights = None

    def fit(self, data, ncomp, sample_log_weights=None,
            features=None, init=None,
            init_label=None, init_para=None,
            epoch=30, update_features_per_epoch=None,
            verbose=True, lock=None):
        "Fit the model with batch EM."
        nsample, dim, nvocab = data.shape
        self.dim, self.nvocab, self.ncomp = dim, nvocab, ncomp

        self._sample_log_weights = sample_log_weights

        self.lock = np.array(lock, dtype=np.bool) if lock is not None else np.zeros((dim, nvocab), dtype=np.bool)

        if features is None:
            self.features = range(dim)
        elif isinstance(features, (list, np.ndarray)):
            self.features = features
        elif isinstance(features, int):
            score = utils.Feature_Selection.MI_score(data, sample_log_weights=sample_log_weights, rm_diag=True, lock=lock)
            sigma = score.sum(axis=1)
            cand = np.argsort(sigma)[::-1]
            self.features = cand[:features]
        else:
            raise ValueError('invalid input type for <features>')

        logger.info(
            "Training an MDPD with dimension %i, %i features, sample size %i, vocab size %i and the target number of components %i",
            self.dim, len(self.features), nsample, self.nvocab, self.ncomp)

        self._model_init(data, init, init_label, init_para)

        self._apply_lock(data)

        self._em_wrapper(data, epoch, update_features_per_epoch, verbose=verbose)

    def _em(self, data):
        """One step EM iteration"""
        # E-step
        log_post = self.log_posterior(data)

        # M-step (full M-step)
        self.logW, self.logC = utils.mstep(log_post, data, sample_log_weights=self._sample_log_weights)
        self._apply_lock(data)

    def _em_wrapper(self, data, epoch, update_features_per_epoch, verbose=False):
        if verbose:
            tmp_folder = tempfile.mkdtemp(dir=self._folder)
            checkpoint_folder = os.path.join(tmp_folder, 'checkpoints')
            os.mkdir(checkpoint_folder)

            self._cache['epoch'] = epoch
            self._cache['update_features_per_epoch'] = update_features_per_epoch
            self._cache['features'].append((0, self.features))

        for ep in xrange(epoch):
            if ep > 0 and update_features_per_epoch is not None and ep % update_features_per_epoch == 0:
                self._update_features(data)

                if verbose:
                    self._cache['features'].append((ep, self.features))

            self._em(data)

            if verbose:
                self._verbose_per_epoch(ep, data)
                self.save(os.path.join(checkpoint_folder, 'epoch_{}'.format(ep)))

        if verbose:
            with open(os.path.join(tmp_folder, 'training_stats.p'), 'w') as h:
                cPickle.dump(self._cache, h)
            logger.info('NOTE: all records and stats are exported to {}'.format(tmp_folder))

    def _verbose_per_epoch(self, ep, data):
        dim = data.shape[1]

        ll = self.log_likelihood(data)
        self._cache['log_likelihood'].append(ll)

        ll_overall = super(MDPD_standard, self).log_likelihood(data)
        self._cache['log_likelihood_overall'].append(ll_overall)

        log_post = self.log_posterior(data)
        score, weighted = utils.Feature_Selection.MI_score_conditional(data, log_post, rm_diag=True)
        sigma_condition = score.sum(axis=1)
        self._cache['sigma_condition'].append(sigma_condition)
        res = np.sum(sigma_condition * weighted[np.newaxis, :]) / (dim * (dim - 1))

        logger.info("iteration %d; log-likelihood (feature selection) %f; "
                    "log_likelihood %f;"
                    "information residue %f",
                    ep,
                    self.log_likelihood(data),
                    super(MDPD_standard, self).log_likelihood(data),
                    res)

    def _model_init(self, data, init, init_label, init_para):
        "initialize logW and logC."
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

    def _apply_lock(self, data):
        "Apply the lock to worker i for the component k, so that i is not discriminant at k."
        # data = data[:, self.features, :]
        lock = self.lock
        if np.any(lock):
            data_weighted = data * self._sample_log_weights[:, None, None]

            log_margin_prob = logsumexp(data_weighted, axis=0) - logsumexp(self._sample_log_weights)    # log(p_0(x_i))
            utils.log_replace_neginf(log_margin_prob)

            newlogC = self.logC.copy()
            lock_broadcast = np.broadcast_to(lock[..., np.newaxis], newlogC.shape)
            newlogC[lock_broadcast] = NINF

            log_margin_prob_sum = logsumexp(log_margin_prob, axis=1, keepdims=True, b=1 - lock)[
                ..., np.newaxis]  # sum_{x_i} log(p_0(x_i)) where are not locked
            newlogC_sum = logsumexp(newlogC, axis=1, keepdims=True)
            newlogC = newlogC - newlogC_sum + log_margin_prob_sum
            newlogC[lock_broadcast] = np.broadcast_to(log_margin_prob[..., np.newaxis], newlogC.shape)[lock_broadcast]

            self.logC = newlogC

    def _update_features(self, data):
        """update features according to conditional information residue"""
        log_post = self.log_posterior(data)
        score, weights = utils.Feature_Selection.MI_score_conditional(data, log_post, rm_diag=True, lock=self.lock)
        sigma = np.sum(score.sum(axis=1) * weights[np.newaxis, :], axis=1)
        cand = np.argsort(sigma)[::-1]
        self.features = cand[:len(self.features)]

    def log_posterior(self, data):
        data_selected = data[:, self.features, :]
        log_joint = utils.log_joint_prob_fast(data_selected, self.logW, self.logC[self.features, ...])
        normalizer = logsumexp(log_joint, axis=1, keepdims=True)
        return log_joint - normalizer

    def log_likelihood(self, data):
        data_selected = data[:, self.features, :]
        log_joint = utils.log_joint_prob_fast(data_selected, self.logW, self.logC[self.features, ...])
        log_marginal_x = logsumexp(log_joint, axis=1)
        return log_marginal_x.mean()

    def MI_residue(self, data):
        log_post = self.log_posterior(data)
        score, weights = utils.Feature_Selection.MI_score_conditional(data, log_post, rm_diag=True, lock=self.lock)
        res = np.sum(score.sum(axis=1) * weights[np.newaxis, :]) / (self.dim * (self.dim - 1))
        logger.info('The mutual information residue (include all features) is {}'.format(res))

        features = np.array(self.features)
        score_select = score[features[:, np.newaxis], features, :]
        res_select = np.sum(score_select.sum(axis=1) * weights[np.newaxis, :]) / (len(features) * (len(features) - 1))
        logger.info('The mutual information residue (within selected features) is {}'.format(res_select))

    def change_features(self, data, features=None):
        """Change the feature set"""
        features = features or range(self.dim)
        self._assert_features(features)
        self.features = features

    @staticmethod
    def _assert_features(features):
        assert not isinstance(features, basestring)

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

    def _tbs(self):
        names = ['features',
                 'lock',
                 '_cache',
                 '_folder',
                 ]

        tbs = super(MDPD_standard, self)._tbs()
        for name in names:
            tbs[name] = getattr(self, name)
        return tbs


class MDPD_online(MDPD_standard):
    """MDPD model with mini-batch EM + Feature Selection!"""
    def __init__(self):
        super(MDPD_online, self).__init__()
        self._log_post = None

    def fit(self, data, ncomp, sample_log_weights=None,
            features=None, init="random",
            init_label=None, init_para=None,
            epoch=30, batch=None, update_features_per_batch=None,
            verbose=True, lock=None):
        "Fit the model with online EM"
        nsample, dim, nvocab = data.shape
        self.dim, self.nvocab, self.ncomp = dim, nvocab, ncomp

        self._sample_log_weights = sample_log_weights

        # lock is not used. All parameters are trainable.
        self.lock = np.zeros((dim, nvocab), dtype=np.bool)

        if features is None:
            self.features = range(dim)
            if update_features_per_batch is not None:
                update_features_per_batch = None
                logger.warn('update_features_per_batch is ignored, because features is not int.')
        elif isinstance(features, (list, np.ndarray)):
            self.features = features
            if update_features_per_batch is not None:
                update_features_per_batch = None
                logger.warn('update_features_per_batch is ignored, because features is not int.')
        elif isinstance(features, int):
            score = utils.Feature_Selection.MI_score(data, sample_log_weights=sample_log_weights, rm_diag=True, lock=lock)
            sigma = score.sum(axis=1)
            cand = np.argsort(sigma)[::-1]
            self.features = cand[:features]
        else:
            raise ValueError('invalid input type for <features>')

        if update_features_per_batch is not None and batch is None:
            update_features_per_batch = None
            logger.warn('update_features_per_batch is ignored, because batch is not set. Batch EM will be used to train the model.')

        logger.info(
            "Training an MDPD with dimension %i, %i features, sample size %i, vocab size %i and the target number of components %i",
            self.dim, len(self.features), nsample, self.nvocab, self.ncomp)

        self._model_init(data, init, init_label, init_para)

        if batch is None:
            # batch EM
            self._em_wrapper(data, epoch, None, verbose=verbose)
        else:
            # Online EM, to trigger online EM, batch (batch_size) is needed
            self._log_post = self.log_posterior(data)
            self._em_online_wrapper(data, epoch, batch, update_features_per_batch, verbose=verbose)

    def _em_online(self, data, batch_index):
        """One step online EM iteration"""
        # partial E-step
        log_post_batch = self.log_posterior(data[batch_index, ...])
        self._log_post[batch_index, ...] = log_post_batch

        # M-step (full M-step)
        self.logW, self.logC = utils.mstep(self._log_post, data, sample_log_weights=self._sample_log_weights)

    def _em_online_wrapper(self, data, epoch, batch, update_features_per_batch, verbose=False):
        nsample = data.shape[0]
        nbatch = int(nsample / batch)
        for ep in xrange(epoch):
            data_idx_rand = np.random.permutation(nsample)
            for nb in xrange(nbatch):
                if update_features_per_batch is not None and nb % update_features_per_batch == 0 and ep + nb > 0:
                    self._update_features(data)
                self._em_online(data, data_idx_rand[nb * batch : (nb + 1) * batch])
            if verbose:
                self._verbose_per_epoch(ep, data)

    def _tbs(self):
        names = ['_log_post']

        tbs = super(MDPD_online, self)._tbs()
        for name in names:
            tbs[name] = getattr(self, name)
        return tbs


class Hierachical_MDPD(object):
    def __init__(self, depth, width=2):
        self.depth = depth
        self.width = width
        total = (width ** (depth + 1) - 1) / (width - 1)
        self.models = [None] * int(total)
        self._debug = [None] * int(total)

    def fit(self, data, features, epoch=100):
        nsample = data.shape[0]
        width, depth = self.width, self.depth

        cache = [(0, - np.log(nsample) * np.ones(nsample))] # index of the model to be trained (for BFS)

        while cache:
            idx, sample_log_weights = cache.pop(0)

            logging.info('Training model ({}, {}) (depth = {})'.format(int(np.log2(idx + 1)), idx + 1 - 2 ** int(np.log2(idx + 1)), self.depth))
            model = self._fit_one_model(data, features, sample_log_weights, epoch)
            self.models[idx] = model
            self._debug[idx] = sample_log_weights

            log_post = model.log_posterior(data)
            log_p_tilde = log_post + sample_log_weights[:, None]
            next_sample_log_weights = log_p_tilde - logsumexp(log_p_tilde, axis=0, keepdims=True)

            if width * idx + width < len(self.models):
                for k in xrange(width):
                    cache.append((width * idx + k + 1, next_sample_log_weights[:, k]))

    def _fit_one_model(self, data, features, sample_log_weights, epoch):
        model = MDPD_standard()
        model.fit(data, self.width, sample_log_weights=sample_log_weights, init='random', features=features, epoch=epoch, verbose=False)
        return model

    def inference_path(self, data):
        "a path of the log weights on data through the model tree."
        nsample = data.shape[0]

        total = int((self.width ** (self.depth + 1) - 1) / (self.width - 1))
        path = [None] * total # log weights on the sample
        path[0] = np.zeros((nsample, 1))

        for idx in xrange(total - self.width ** self.depth):
            log_joint = path[idx]

            model = self.models[idx]
            log_post = model.log_posterior(data)
            new_log_joint = log_post + log_joint

            for k in xrange(self.width):
                child_idx = idx * self.width + k + 1
                path[child_idx] = new_log_joint[:, k][:, None]

        return path

    def log_posterior(self, data):
        "Inference of the hierachical MDPD. (BFS)"
        path = self.inference_path(data)

        cache = []

        total = len(self.models)
        n_leafs = self.width ** self.depth

        for i, log_joint in enumerate(path[-n_leafs:]):
            idx = total - n_leafs + i

            model = self.models[idx]
            log_post = model.log_posterior(data)
            new_log_joint = log_post + log_joint

            cache.append(new_log_joint)

        log_post = np.concatenate(cache, axis=1)
        log_post = log_post - logsumexp(log_post, axis=1, keepdims=True)
        return log_post

    @property
    def leaf_nodes(self):
        return self.models[- (self.width**self.depth):]

    def _tbs(self):
        names = ['depth',
                 'width',
                 '_debug'
                 ]

        tbs = {}
        for name in names:
            tbs[name] = getattr(self, name)

        return tbs

    def save(self, filename):
        cache = self._tbs()

        cache['models'] = [model._tbs() for model in self.models]

        with open(filename, 'wb') as f:
            cPickle.dump(cache, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            cache = cPickle.load(f)

        def create_model(tbl):
            model = MDPD_standard()
            model.load_dic(tbl)
            return model

        for key, val in cache.iteritems():
            if hasattr(self, key):
                if key == 'models':
                    self.models = [create_model(tbl) for tbl in cache['models']]
                else:
                    setattr(self, key, val)


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





