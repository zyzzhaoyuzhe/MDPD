"""
Miscellaneous functions regarding MDPD instances.
"""


import sys
import warnings
import copy
from distance import *


def align(model1, model2):
    if model1.c != model2.c:
        warnings.warn('Mismatch: number of components.')
        return model2
    order = []
    candidate = range(model2.c)
    for k1 in range(model1.c):
        foo = sys.float_info.max
        for k2 in range(len(candidate)):
            bar = js(model1.C, k1, model2.C, candidate[k2])
            if bar < foo:
                idx = k2
                foo = bar
        order.append(candidate.pop(idx))
    # swap
    if np.unique(order).size == model1.c:
        output = copy.deepcopy(model2)
        output.reorder(order)
        return output
    else:
        return model2


def contribution(W, C):
    """
    Return a matrix of f(Y|X_i)
    :param W:
    :param C:
    :return:
    """
    output = []
    m = len(C)
    for mat in C:
        c = W.shape[0]
        foo = W.reshape((1, c))
        foo = mat*foo   # f(x_i,y)
        output.append(foo/np.sum(foo, axis=1).reshape(foo.shape[0], 1))
    return output


def rm_diag(mat):
    cp = copy.deepcopy(mat)
    if isinstance(mat, np.ndarray):
        if len(mat.shape) == 2:
            n = min(mat.shape)
            for i in range(n):
                cp[i, i] = 0
        elif len(mat.shape) == 3:
            n = min(mat.shape[:2])
            for i in range(n):
                cp[i, i, :] = 0
    return cp
