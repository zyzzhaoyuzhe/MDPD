
"""
Experiments of diagonal dominance case
"""

import time
import scipy.io as scio
import matplotlib.pyplot as plt
from MDPD import *
import numpy as np
from multiprocessing import Pool
import sys
from MDPD.utils import mylog
import cPickle
import multiprocessing

ncpu = multiprocessing.cpu_count()

num_component = int(sys.argv[1])
# num_component = 2

m = 100
n = 1000

gap = 0.1

nran = 10
low = np.linspace(1./num_component + 0.2, 1./num_component, num=nran)
high = low + gap

range_list = []
for i in range(nran):
    range_list.append((low[i], high[i]))

n_epoch = 10
n_iter = 40

def one_epoch(ran):
    """

    :return:
    """
    print ran
    # truemodel
    print "benchmark"
    truemodel = MDPD.MDPD()
    truemodel.get_config((m, n, num_component, num_component))
    W, C = model_gen.crowdsourcing_rand(m, num_component, ran)
    truemodel.get_modelgen(W, C)
    truemodel.copy_CW(W, C)
    train, label = truemodel.gen_Data(n)

    benchmark = {}
    benchmark['L'] = truemodel.log_likelihood(train)
    benchmark['MI'] = truemodel.get_MI(train, rm_diag=True)
    benchmark['post'] = truemodel.log_posterior(train, subset=range(truemodel.m))
    benchmark['predict'] = truemodel.predict(train, subset=range(truemodel.m), label=label)[1]

    # spectral method
    print "spectral method"
    try:
        model_spec = MDPD.MDPD()
        model_spec.get_config((m, n, num_component, num_component))
        rec = model_spec.fit(train, init='spectral', track=True, display=False)
        model_spec = misc.align(truemodel, model_spec)

        spectral = {}
        spectral['record'] = rec
        spectral['predict'] = model_spec.predict(train, subset=range(model_spec.m), label=label)[1]
    except:
        spectral = None

    # majority vote EM
    print "majority vote EM"
    model_mv = MDPD.MDPD()
    model_mv.get_config((m, n, num_component, num_component))
    rec = model_mv.fit(train, init='majority', stopcrit='number of iterations', epoch=n_iter, track=True, display=False)
    model_mv = misc.align(truemodel, model_mv)

    majority = {}
    majority['record'] = rec
    majority['predict'] = model_mv.predict(train, subset=range(model_mv.m), label=label)[1]

    # stageEM
    print "StageEM"
    try:
        model_stage = MDPD.MDPD()
        model_stage.get_config((m, n, num_component, num_component))
        rec = model_stage.fit(train, init='StageEM', stopcrit='number of iterations', epoch=n_iter, track=True, display=False)
        model_stage = misc.align(truemodel, model_stage)

        stageEM = {}
        stageEM['record'] = rec
        stageEM['predict'] = model_stage.predict(train, subset=model_stage.activeset, label=label)[1]

        model_stage.refine(train)
        stageEM['L_refined'] = model_stage.log_likelihood(train)
        stageEM['predict_refine'] = model_stage.predict(train, subset=range(model_stage.m), label=label)[1]
    except:
        print "stageEM failed"
        stageEM = None

    return benchmark, spectral, majority, stageEM

def multi_epoch(ran):
    epoch = []
    for i in range(n_epoch):
        epoch.append(one_epoch(ran))
    return epoch

def test(x):
    time.sleep(10)
    return x

tbs = {}
tbs['parameter'] = range_list

pool = Pool(processes=ncpu)
output = pool.map(one_epoch, range_list)

tbs['results'] = output


cPickle.dump(tbs, open('EXP_diag_'+str(num_component)+'.p', 'wb'))

print 1
