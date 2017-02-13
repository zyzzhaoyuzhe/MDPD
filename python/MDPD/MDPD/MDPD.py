
"""
The core module of the package.

Basic MDPD class is defined in this module which is composed of basic functions. Stage-wise EM is defined as a subclass of MDPD.

"""
from __future__ import division
from helper import *
from tensor_power import *
import misc
import random
import math
import numpy as np
import cPickle
import copy
import logging
from collections import defaultdict
from scipy.stats import mode

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(10)


class MDPD_basic(object):
    """
    Basic MDPD class.

    Key attributes:
    :param m: Dimension (scalar)
    :param n: Number of samples (scalar)
    :param c: Number of mixture components (scalar)
    :param r: Size of discrete output alphabet (array: m)
    :param z: Samples (m-r-n)
    :param label: True labels of samples (n)
    :param W: Mixing weights (c)
    :param C: Confusion Matrix (m-r-c)
    :param Wgen: Generating model mixting weights (c)
    :param Cgen: Generating model confusion matrix (m-r-c)
    :param trace: to track the merge history (list: m)
    """
    def __init__(self):
        '''constructor'''
        self.dim = None  # dimension
        self.trace = None   # trace used by merge method
        self.nsample = None  # number of sample
        self.ncomp = None  # number of components (or the target number of components)
        self.nvocab = None  # size of discrete alphabet
        self.label = None
        self.logW = None  # mixing weights
        self.logC = None  # Confusion matrix

    def initCW(self):
        '''Initialize W and C, assuming m, n, r, c were set.'''
        self.logC = np.ones([self.dim, self.nvocab, self.ncomp], dtype=np.float)
        self.logW = np.ones(self.ncomp, dtype=np.float)

    def get_config(self, dim=None, nsample=None, ncomp=None, nvocab=None):
        """Get model configuration"""
        self.dim = int(dim)
        self.nsample = int(nsample)
        self.ncomp = int(ncomp)
        self.nvocab = int(nvocab)

    def copy_CW(self, logW, logC):
        """Get model parameters."""
        self.logC = copy.deepcopy(logC)
        self.logW = copy.deepcopy(logW)

    def loglikelihood(self, data):
        foo = log_comL_fast(self.nsample, data, self.logW, self.logC)
        lsum = logsumexp(foo, axis=0)
        return lsum.mean()

    def logposterior(self, data):
        foo = log_comL_fast(self.nsample, data, self.logW, self.logC)
        lsum = logsumexp(foo, axis=0)
        return foo - lsum

    def predict(self, data, label=None):
        if label is None:
            label = self.label
        logpost = self.logposterior(data=data)
        out = np.argmax(logpost, axis=0)
        if label.shape != out.shape:
            return out
        else:
            n = label.size
            err = sum(out - label != 0) / n
            print 'error rate: {0:.2%}'.format(err)
            return out, err

    def swap_Label(self, order):
        """Perform label swap according to 'order'."""
        self.logW = self.logW[order]
        self.logC = self.logC[:, :, order]

    def _get_tbs(self):
        """helper for save"""
        tbs = {}  # to be saved
        tbs['m'] = self.dim
        tbs['n'] = self.nsample
        tbs['c'] = self.ncomp
        tbs['r'] = self.nvocab
        tbs['z'] = self.data
        tbs['label'] = self.label
        tbs['W'] = self.logW
        tbs['C'] = self.logC
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

        with open(filename,'wb') as f:
            cPickle.dump(tbs, f)

    def _load(self, tbl):
        """Helper for load"""
        self.dim = tbl['m']
        self.nsample = tbl['n']
        self.ncomp = tbl['c']
        self.nvocab = tbl['r']
        self.data = tbl['z']
        self.label = tbl['label']
        self.logW = tbl['W']
        self.logC = tbl['C']
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
        with open(filename,'rb') as f:
            tbl = cPickle.load(f)
            self._load(tbl)

    def merge(self, cord_list):
        _num = len(cord_list)
        newm = self.dim - _num + 1
        newr = 1
        newtrace = []
        foo = []    # for z
        bar = [1]
        for i in cord_list:
            newr *= self.nvocab[i]
            newtrace += self.trace[i]
            # binary coding to numerics
            foo.append(np.dot(np.asarray(range(self.nvocab[i])), self.data[i]))
            bar.append(bar[-1] * self.nvocab[i])
        bar.pop()
        bar = np.asarray(bar).reshape((_num, 1))
        foo = np.asarray(foo)   # numeric responses of each in cord_list
        numeric_res = np.sum(foo * bar, axis=0) # numeric responses to each sample for the merged coordinate
        # convert to binary encoding
        foo = numeric_res * self.nsample + np.asarray(range(self.nsample))
        newz = np.zeros(newr * self.nsample)
        newz[foo] = 1
        newz = newz.reshape((newr, self.nsample))
        # newC and newCgen
        newC = np.zeros((newr, self.ncomp))
        if self.Cgen:
            newCgen = np.zeros((newr, self.ncomp))
        for k in range(self.ncomp):
            cord_list_copy = copy.copy(cord_list)
            id = cord_list_copy.pop()
            foo1 = self.logC[id][:, k].flatten()
            if self.Cgen:
                foo2 = self.Cgen[id][:, k].flatten()
            for count in range(1, _num):
                id = cord_list_copy.pop()
                foo1 = np.outer(foo1, self.logC[id][:, k]).flatten()
                if self.Cgen:
                    foo2 = np.outer(foo2, self.logC[id][:, k]).flatten()
            newC[:, k] = foo1
            if self.Cgen:
                newCgen[:, k] = foo2
        # clean up model
        self.dim = newm
        self.nvocab.append(newr)
        self.nvocab[:] = [item for i, item in enumerate(self.nvocab) if i not in cord_list]
        self.data.append(newz)
        self.data[:] = [item for i, item in enumerate(self.data) if i not in cord_list]
        self.logC.append(newC)
        self.logC = [item for i, item in enumerate(self.logC) if i not in cord_list]
        if self.Cgen:
            self.Cgen.append(newCgen)
            self.Cgen = [item for i, item in enumerate(self.Cgen) if i not in cord_list]
        self.trace.append(newtrace)
        self.trace = [item for i, item in enumerate(self.trace) if i not in cord_list]


class MDPD(MDPD_basic, object):
    """
    A subclass of MDPD.

    Key parameter:
    :param: activeset
    :param: c_target
    """
    def __init__(self, data, ncomp):
        MDPD_basic.__init__(self)
        nsample, dim, nvocab = data.shape
        self.nsample = nsample
        self.dim = dim
        self.nvocab = nvocab
        self.ncomp = ncomp
        ## core attributes
        self.infoset = []
        self.MIres = None
        ## temporary attributes
        self.c_target = None
        self.lock= None

    def train(self, data, ncomp=None, method="StageEM", niter=30, diffthre=0.001, num_component=None, reset=True,
              display=True):
        logger.info("Training an MDPD with dimension %i, sample size %i, vocab size %i and the target number of components %i",
                    self.dim, self.nsample, self.nvocab, self.ncomp)
        if reset:
            logger.info("All weights will be re-initialized.")

        num_component = num_component or self.ncomp
        # choosing fitting methods
        if method == "StageEM": # stagewise EM
            output = defaultdict(list)
            # target number of components
            if ncomp is None:
                c_target = self.ncomp
            else:
                c_target = ncomp
            # initialize the model from scratch
            if reset:
                self.reset(data)
            # statistics
            ll = self.loglikelihood(data)
            MI_t = self.get_MI(data, rm_diag=True)
            #
            output['ll'].append(ll)
            output['MI'].append(MI_t)
            output['len_active'].append(0)
            output['post'].append(self.logposterior(data))
            # iteration initialization
            count = 0
            # body
            while True:
                count += 1
                ### algorithm starts
                # update MIres
                # MIres = self.get_MIres(data, rm_diag=True)
                MIres = self.get_MIres(data, rm_diag=True, wPMI=True)
                # find the biggest entry
                cord1, cord2, comp = np.unravel_index(MIres.argmax(), MIres.shape)
                # update infoset
                if cord1 not in self.infoset or cord2 not in self.infoset:
                    self.add_infoset(cord1)
                    self.add_infoset(cord2)
                    # split component comp at cord1 and cord2
                    if self.ncomp < c_target:
                        self.split(data, cord1, cord2, comp)
                        output['split_history'].append((cord1, cord2, comp))
                # EM with activeset
                self.EM(data, self.infoset)
                ### algorithm finishes
                # measure performance
                ll = self.loglikelihood(data)
                MI_t = self.get_MI(data, rm_diag=True)
                if display:
                    logger.info("iteration %d; log-likelihood %f; MI.max %f; # active set %d; # component %d", count, ll, MI_t.max(), len(self.infoset), self.ncomp)
                output['ll'].append(ll)
                output['MI'].append(MI_t)
                output['len_active'].append(len(self.infoset))
                output['post'].append(self.logposterior(data, subset=self.infoset))
                # check stop criterion
                if count >= niter:
                    break
            return output
        elif method in ["majority", "plain", "spectral"]:  # use majority vote to initialize EM or plain EM
            if method == "majority":
                self.init_mv(data)
            elif method == "plain":
                self.init_random()
            elif method == "spectral":
                self.init_spec(data)
                # run 1 runs of EM after initialization
                niter = 2

            # statistics
            ll = self.loglikelihood(data)
            #
            # if track:
            #     rec_L = [ll]
            #     rec_post = [self.posterior(data, subset=range(self.dim))]
            # iteration initialization
            count = 0
            while True:
                count += 1
                self.EM(data, range(self.dim))
                ll = self.loglikelihood(data)
                # whether to display
                if display:
                    logger.info("iteration %d; log-likelihood %f;", count, ll)
                if count >= niter:
                    break

    def reset(self, data):
        self.infoset = []
        self.logC = np.log(data.mean(axis=0).squeeze()[:, :, np.newaxis])
        if self.lock is None:
            self.lock = np.ones(self.logC.shape)
        # self.C = [data[i].mean(axis=1)[:, np.newaxis] for i in range(self.dim)]
        self.logW = np.asarray([0])
        self.ncomp = 1
        pass

    def init_random(self): # random init
        self.logW = np.ones(self.ncomp) / self.ncomp
        self.logC = np.random.random_sample((self.dim, self.nvocab, self.ncomp))
        self.logC /= self.logC.sum(axis=1)[:, np.newaxis, :]
        if self.lock is None:
            self.lock = np.zeros(self.logC.shape)

    def init_mv(self, data): # majority vote
        """

        :return:
        """
        # make sure all workers has same output size
        post = np.mean(data, axis=1).T
        W, C = self.Mstep(post, data)
        self.logW = W
        self.logC = C
        if self.lock is None:
            self.lock = np.zeros(self.logC.shape)

    def init_spec(self, data):
        """
        Use spectral methods to initialize EM. (Zhang. and Jordan. 2014)
        :param data:
        :return:
        """
        # divide workers into 3 partitions
        num_worker_partition = int(self.dim / 3)
        foo = np.random.permutation(self.dim)

        partition = [None] * 3
        partition[0] = foo[:num_worker_partition]
        partition[1] = foo[num_worker_partition:2*num_worker_partition]
        partition[2] = foo[2*num_worker_partition:]

        # calculate average response to each sample for each group
        train_g = [None]*3
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
            m2, m3 = get_tensor(train_g, a, b, c)
            foo, bar = tensorpower(m2, m3, self.ncomp)
            W.append(foo)
            Cg.append(bar)
        self.logW = np.mean(np.asarray(W), axis=0)
        # use Cg to recover C for each worker
        C = np.zeros((self.dim, self.nvocab, self.ncomp))
        for i in range(self.dim):
            foo = np.zeros((self.ncomp, self.ncomp))
            for g in range(3):
                if i not in partition[g]:
                    foo += get_Ci(train_g, self.logW, Cg, g, data, i)
            foo = foo/2
            C[i, :, :] = foo
            # C.append(foo)
        self.logC = C

    # fine tune
    def refine(self, data, infoset=None, niter=20):
        logger.info('Fine tune the model with ')
        infoset = infoset or range(self.dim)
        for count in range(niter):
            self.EM(data, infoset)

    # split component
    def split(self, data, cord1, cord2, comp):
        #
        lock1 = self.lock[cord1, :, comp]
        lock2 = self.lock[cord2, :, comp]
        free1 = np.where(lock1 == 0)[0]
        free2 = np.where(lock2 == 0)[0]

        # get Hessian (Hessian of marginal loglikelihood)
        H = get_Hessian2(data, self.logW, self.logC, self.lock, self.infoset, cord1, cord2, comp)
        # solve constrained singular value problem
        U, Sig, V = np.linalg.svd(H)
        V = V.T
        m, n = U.shape
        allone = np.ones(m) * 1 / np.sqrt(m)
        for i in xrange(n):
            U[:, i] -= np.dot(U[:, i], allone) * allone
        m, n = V.shape
        allone = np.ones(m) * 1 / np.sqrt(m)
        for i in xrange(n):
            V[:, i] -= np.dot(V[:, i], allone) * allone

        H_star = U.dot(np.diag(Sig)).dot(V.T)
        U, Sig, V = np.linalg.svd(H_star)
        V = V.T

        #
        dp1 = U[:,0]
        dp2 = V[:,0]

        # split component k into two
        self.logW, self.logC, self.ncomp = comp_duplicate(self.logW, self.logC, comp)
        self.lock = np.append(self.lock, self.lock[:, :, comp][:, :, np.newaxis], axis=2)
        ## break symmetry
        stepsize = .05
        def add(logC, lock, cord, comp, dp):
            foo = np.exp(logC[cord, lock[cord, :, comp]==1, comp]) + dp
            foo[foo<0] = 0.001
            logC[cord, lock[cord, :, comp]==1, comp] = np.log(foo)

        add(self.logC, self.lock, cord1, comp, dp1*stepsize)
        add(self.logC, self.lock, cord1, -1, -dp1 * stepsize)
        add(self.logC, self.lock, cord2, comp, dp2 * stepsize)
        add(self.logC, self.lock, cord1, -1, -dp2 * stepsize)

    ## EM step
    def EM(self, data, subset):
        #
        lock = self.lock
        # E-step
        post = self.logposterior(data, subset=subset)
        # M-step (full M-step)
        newlogW, newlogC = self.Mstep(post, data)
        if np.any(lock==0):
            newlogC[lock==0] = -100
            self.logC[lock==1] = -100
            newlogC = newlogC - logsumexp(newlogC, axis=1)[:, np.newaxis, :] \
                      + np.log(1 - np.exp(logsumexp(self.logC, axis=1)))[:, np.newaxis, :]
            newlogC[lock==0] = self.logC[lock==0]
        self.logW = newlogW
        self.logC = newlogC

    ## M step
    def Mstep(self, post, data):
        return Mstep(post, data)

    ## get mutual inforamtion conditional on component (m-m-n-c)
    def get_MIres(self, data, rm_diag=False, wPMI=False):
        return get_MIres_fast(data, self.logW, self.logC, self.infoset, rm_diag=rm_diag, wPMI=wPMI)

    ## get MI
    def get_MI(self, data, rm_diag=False, subset=None):
        return get_MI(data, self.logW, self.logC, subset or self.infoset, rm_diag=rm_diag)

    ## posterior (overwrite)
    def logposterior(self, data, subset=None):
        return logposterior_StageEM(data, self.logW, self.logC, subset or self.infoset)

    ## predict (overwrite)
    def predict(self, data, label, subset=None):
        logpost = self.logposterior(data, subset=subset or self.infoset)
        out = np.argmax(logpost, axis=0)
        if label.shape != out.shape:
            return out
        else:
            n = label.size
            err = sum(out - label != 0)/n
            print 'error rate: {0:.2%}'.format(err)
            return out, err

    ## Swap
    def swap(self, idx1, idx2):
        foo = self.logW[idx1]
        self.logW[idx1] = self.logW[idx2]
        foo = np.copy(self.logC[:, :, idx1])
        self.logC[:, :, idx1] = self.logC[:, :, idx2]
        self.logC[:, :, idx2] = foo

    def align(self, data, label, subset):
        pred, err = self.predict(data, label, subset=subset)
        order = []
        for k in range(self.ncomp):
            u, freq = np.unique(pred[label==k], return_counts=True)
            order.append(u[freq.argmax()])
        # print order
        self.swap_Label(order)

    def add_infoset(self, input):
        self.infoset = add_infoset(self.infoset, input)

    def save(self, filename):
        """override save"""
        if filename[-2:] != '.p':
            filename += '.p'
        tbs = self._get_tbs()
        tbs['activeset'] = self.infoset
        with open(filename, 'wb') as f:
            cPickle.dump(tbs, f)

    def load(self, filename):
        """override load"""
        if filename[-2:] != '.p':
            filename += '.p'
        with open(filename, 'rb') as f:
            tbl = cPickle.load(f)
            self._load(tbl)
            self.infoset = tbl['activeset']

    def merge(self, cord_list):
        """

        :param cord_list:
        :return:
        """
        _num = len(cord_list)
        newm = self.dim - _num + 1
        idx = [item for i, item in enumerate(range(self.dim)) if i not in cord_list]
        newactiveset = []
        for i in self.infoset:
            if i not in cord_list:
                newactiveset.append(idx.index(i))
            else:
                if newm - 1 not in newactiveset:
                    newactiveset.append(newm - 1)
        self.infoset = newactiveset
        MDPD_basic.merge(self, cord_list)

