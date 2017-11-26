'''
This module contains methods to generate specific MDPD models.
'''

import numpy as np
import sys
import warnings

def crowdsourcing_rand(dim, ncomp, rang):
    '''

        :param dim:
        :param ncomp:
        :param rang:
        :return:
        '''
    Wgen = np.ones(ncomp) / float(ncomp)
    #
    probhi = rang[1]
    problow = rang[0]
    # check if problow is larger than 1/c
    if problow < 1. / ncomp:
        warnings.warn('Warning: the diagonal of confusion matrix may not dominate.')
    _diag = problow + (probhi - problow) * np.random.random_sample((ncomp, dim))
    baz = np.random.random_sample((ncomp - 1, dim, ncomp))
    Cgen = []
    for i in range(dim):
        foo = np.diag(_diag[:, i])
        res = 1 - foo.sum(axis=1)
        res = np.squeeze(res)
        for k in range(ncomp):
            _res = res[k]
            # set off diagonal entries for component k
            tmp = baz[:, i, k]
            bar = tmp/np.sum(tmp) * _res
            # plug bar into foo
            mask = np.asarray(range(ncomp))
            mask = mask != k
            foo[mask, k] = bar
        Cgen.append(foo)
    return Wgen, Cgen


def crowdsourcing_diag(m, c, ran):
    '''

    :param m:
    :param c:
    :param ran:
    :return:
    '''
    Wgen = np.ones(c) / float(c)
    #
    probhi = ran[1]
    problow = ran[0]
    # check if problow is larger than 1/c
    if problow < 1./c:
        warnings.warn('Warning: the diagonal of confusion matrix may not dominate.')
    _diag = problow + (probhi - problow) * np.random.random_sample((c, m))
    Cgen = []
    for i in range(m):
        foo = np.diag(_diag[:, i])
        res = 1 - foo.sum(axis=1)
        res = np.squeeze(res)
        for k in range(c):
            _res = res[k]
            # set off diagonal entries for component k
            bar = np.ones(c-1) * _res/(c-1)
            # plug bar into foo
            mask = np.asarray(range(c))
            mask = mask != k
            foo[mask, k] = bar
        Cgen.append(foo)
    return Wgen, Cgen


def crowdsoucing_sparse(m, c, sparsity, prob_diag):
    '''

    :param m:
    :param c:
    :param sparsity:
    :param prob_diag:
    :param prob_rand:
    :return:
    '''
    idiot_low = 0.1
    idiot_high = 0.9

    num_expert = int(m * sparsity)

    diag = np.linspace(prob_diag[1], prob_diag[0], num=num_expert)
    Wgen = np.ones(c)*1./c
    Cgen = []
    for i in range(num_expert):
        res = 1 - diag[i]
        foo = np.ones((c, c)) * diag[i]
        bar = np.ones(c-1) * res/(c-1)
        for k in range(c):
            mask = np.asarray(range(c))
            mask = mask != k
            foo[mask, k] = bar
        Cgen.append(foo)


    # Wgen, Cgen = crowdsourcing_diag(num_expert, c, prob_diag)

    num_rand = m - num_expert
    foo = idiot_low + np.random.random_sample((c, num_rand)) * (idiot_high - idiot_low)
    for i in range(num_rand):
        bar = foo[:, [i]]
        bar = bar/np.sum(bar)
        bar = bar.repeat(c, axis=1)
        Cgen.append(bar)
    return Wgen, Cgen
