#!/usr/bin/python

import time
import scipy.io as scio
import matplotlib.pyplot as plt
from MDPD import *
import numpy as np
from multiprocessing import Pool
import sys
from MDPD.utils import mylog
import cPickle

low = sys.argv[1]
high = sys.argv[2]
low = float(low)
high = float(high)
ran = (low,high)
print ran

m = 100
n = 1000
c = 3

nmodel = 20
problow = np.linspace(0.6, 0.34, num=nmodel)
probhi = problow + 0.1
inputs = []
for i in range(nmodel):
    inputs.append((problow[i], probhi[i]))

def one_model(num_comp):
    print 'number of components' + str(num_comp)
    model = MDPD.StageEM()
    model.get_config(m, n, num_comp, c)
    model.read(truemodel.z, label=truemodel.label)
    foo = model.learn(stopcrit='number of iterations', num_iter=30, stats_output=True, display=False)
    return foo, model

def real_MI(model):
    output = np.zeros((model.m, model.m))
    for i in range(model.m):
        Ci = np.dot(model.C[i], model.W)
        for j in range(model.m):
            Cj = np.dot(model.C[j], model.W)
            down = np.outer(Ci, Cj)
            foo = np.einsum('i..., j...', model.C[i], model.C[j])
            up = np.einsum('k..., k', foo, model.W)
            output[i, j] = np.sum(up * mylog(up / down))
    return output

if __name__ == '__main__':
    truemodel = MDPD.StageEM()
    truemodel.get_config(m, n, c, c)
    W, C = model_gen.crowdsourcing_rand(m, c, ran)
    truemodel.get_modelgen(W, C)
    truemodel.copy_CW(W, C)
    truemodel.gen_Data(n)
    disp.comp_heatmap(truemodel.C)

    inputs = [2, 3, 4]
    p = Pool(processes=3)
    outputs = p.map(one_model, inputs)
    print 'model training complete'
    for i in range(3):
        disp.comp_heatmap(outputs[i][1].C)
        disp.comp_heatmap(outputs[i][1].C, list=outputs[i][1].activeset[:10])
        disp.show_rec(outputs[i][0])
        print outputs[i][1].activeset
    plt.show()
    raw_input()