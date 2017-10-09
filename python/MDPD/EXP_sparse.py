
"""
Experiments of sparse
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

num_component = int(sys.argv[1])
# num_component = 2

m = 100
n = 1000

n_sparse = 10
sparse_list = np.linspace(0.20, 0.05, num=n_sparse)
if num_component == 2:
    problow = 0.6
    probhigh = 0.8
elif num_component == 3:
    problow = 0.6
    probhigh = 0.6
elif num_component == 4:
    problow = 0.35
    probhigh = 0.55

n_epoch = 10
n_iter = 40

def one_epoch(sparse):
    """

    :return:
    """
    # truemodel
    print "true model (benchmark)"
    truemodel = MDPD.MDPD()
    truemodel.get_config((m, n, num_component, num_component))
    W, C = model_gen.crowdsoucing_sparse(m, num_component, sparse, [problow, probhigh])

    print C[0]

    truemodel.get_modelgen(W, C)
    truemodel.copy_CW(W, C)
    train, label = truemodel.gen_Data(n)

    benchmark = {}
    benchmark['L'] = truemodel.log_likelihood(train)
    benchmark['MI'] = truemodel.get_MI(train, rm_diag=True)
    benchmark['post'] = truemodel.log_posterior(train, subset=range(truemodel.m))
    benchmark['predict'] = truemodel.predict(train, subset=range(truemodel.m), label=label)[1]

    # spectral method
    print "spectral Method"
    try:
        model_spec = MDPD.MDPD()
        model_spec.get_config((m, n, num_component, num_component))
        rec = model_spec.train(train, method='spectral', track=True, display=False)
        model_spec = misc.align(truemodel, model_spec)

        spectral = {}
        spectral['record'] = rec
        spectral['predict'] = model_spec.predict(train, subset=range(model_spec.m), label=label)[1]
    except:
        print "Spectral Method failed."
        spectral = None

    # majority vote EM
    print "majority Vote EM"
    model_mv = MDPD.MDPD()
    model_mv.get_config((m, n, num_component, num_component))
    rec = model_mv.train(train, method='majority', stopcrit='number of iterations', niter=n_iter, track=True, display=False)
    model_mv = misc.align(truemodel, model_mv)

    majority = {}
    majority['record'] = rec
    majority['predict'] = model_mv.predict(train, subset=range(model_mv.m), label=label)[1]

    # stageEM
    print "Stagewise EM"
    try:
        model_stage = MDPD.MDPD()
        model_stage.get_config((m, n, num_component, num_component))
        rec = model_stage.train(train, method='StageEM', stopcrit='number of iterations', niter=n_iter, track=True, display=False)
        model_stage = misc.align(truemodel, model_stage)

        stageEM = {}
        stageEM['record'] = rec
        stageEM['predict'] = model_stage.predict(train, subset=model_stage.activeset, label=label)[1]

        model_stage.refine(train)
        stageEM['L_refined'] = model_stage.log_likelihood(train)
        stageEM['predict_refine'] = model_stage.predict(train, subset=range(model_stage.m), label=label)[1]
    except:
        print "Stagewise EM failed"
        stageEM = None

    return benchmark, spectral, majority, stageEM

def multi_epoch(sparse):
    print sparse
    epoch = []
    for i in range(n_epoch):
        epoch.append(one_epoch(sparse))
    return epoch

def test(x):
    # time.sleep(10)
    W, C = model_gen.crowdsoucing_sparse(100, 2, 0.2, [0.6, 0.7])
    print C[50]
    return np.random.random_sample(1)

tbs = {}
tbs['parameter'] = sparse_list

pool = Pool()
output = pool.map(multi_epoch, sparse_list)
pool.close()
# bar = one_epoch(sparse_list[count])

tbs['results'] = output


cPickle.dump(tbs, open('EXP_sparse_'+str(num_component)+'.p', 'wb'))

print 1

