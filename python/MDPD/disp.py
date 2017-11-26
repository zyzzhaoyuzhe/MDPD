"""
Display methods for MDPD objects
"""
from __future__ import division
from utils import *
import random
import numpy as np
import cPickle
import copy
import misc
import matplotlib.pyplot as plt
import math


def tuning_curve_indi(model, i, res_range=None):
    y = model.predict(None)
    label_y = tuning_curve_comp(model.label, y, disp=False)
    contri = misc.contribution(model.W, model.C)
    contri = contri[i]
    n = contri.shape[0]
    plt.figure()
    if res_range is None:
        res_range = range(n)
    elif isinstance(res_range, int):
        res_range = [res_range]
    for res in res_range:
        foo = np.dot(contri[res, :], label_y) * np.dot(model.C[i][res, :], model.W)
        plt.plot(foo, label=str(res))
    plt.legend(loc=0)
    plt.show(block=False)


def tuning_curve_comp(label, y, disp=True):
    """
    Label can be any attribute associated with data and output_predict is the hidden variable associated with data. The method outputs the distribution of label conditional on the hidden variable.
    :param label: assume to be discrete.
    :param model:
    :return:
    """
    foo = np.unique(y)
    bins = np.unique(label).size
    label_y = []
    #
    c = foo.size
    ncol = int(math.ceil(math.sqrt(c)))
    nrow = int(math.ceil(c/float(ncol)))
    if disp:
        plt.figure()
    for n, item in enumerate(foo):
        if disp:
            plt.subplot(nrow, ncol, n+1)
        bar = np.histogram(label[y == item], bins=bins)
        if disp:
            plt.hist(label[y == item], bins=bins)
        label_y.append(bar[0]/float(bar[0].sum()))
    if disp:
        plt.show(block=False)
    return np.asarray(label_y)


def show_rec(history):
    """Show the record of training"""
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('log-likelihood')
    plt.plot(history['loglikelihood'])
    plt.subplot(1, 3, 2)
    plt.title('Max mutual information')
    plt.plot([item.max() for item in history['MI']])
    plt.subplot(1, 3, 3)
    plt.title('size of the activeset')
    plt.plot(history['len_activeset'])
    plt.show(block=False)


def comp_heatmap(C, list=None):
    """Show each component for each coordinate with heatmap"""
    m = len(C)
    c = C[0].shape[1]
    if list is None:
        list = range(min([m, 10]))
    list_copy = copy.deepcopy(list)

    heat = C[list_copy.pop(0)]
    for i in list_copy:
        heat = np.append(heat, np.zeros((1, c)), axis=0)
        heat = np.append(heat, C[i], axis=0)
    plt.figure()
    plt.title(str(list))
    plt.pcolor(np.flipud(heat), cmap=plt.cm.Blues, alpha=0.8)
    plt.show(block=False)