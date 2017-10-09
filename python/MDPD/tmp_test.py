import time
import scipy.io as scio
from MDPD import *
import numpy as np
from multiprocessing import Pool
import sys
import matplotlib.pyplot as plt
import cPickle

m = 100
n = 1000
c = 3

truemodel = MDPD.MDPD()
truemodel.get_config((m, n, c, c))

# W, C = model_gen.crowdsourcing_rand(m, c, (0.15, 0.35))
W, C = model_gen.crowdsoucing_sparse(m, c, 0.30, [0.45, 0.7])

truemodel.get_modelgen(W, C)
truemodel.copy_CW(W, C)
train, label = truemodel.gen_Data(n)
truemodel.predict(train, subset=range(truemodel.m), label=label)
print truemodel.log_likelihood(train)
# disp.comp_heatmap(truemodel.C)
MI = truemodel.get_MI(train,rm_diag=True,subset=range(30))
print 'benchmark MI'
print np.max(MI)
print np.min(MI+np.eye(100))


print "\nStage EM"
SEM = MDPD.MDPD()
SEM.get_config((m, n, c, c))
rec = SEM.train(train, method='StageEM', stopcrit='number of iterations', niter=20, track=True, display=True)
SEM = misc.align(truemodel, SEM)

disp.show_rec(rec)

SEM.predict(train, subset=SEM.activeset, label=label)
print SEM.activeset
# print foo[label==0][:20]
# print foo[label==1][:20]
SEM.refine(train)
SEM.predict(train, subset=range(SEM.m), label=label)
print SEM.log_likelihood(train)

print "\nmajority vote EM"
foo = np.array(train)
post = np.mean(foo, axis=0)
out = np.argmax(post, axis=0)
n = label.size
err = sum(out - label != 0) / float(n)
print 'MV only error rate: {0:.2%}'.format(err)


model_majority = MDPD.MDPD()
model_majority.get_config((m, n, c, c))
foo = model_majority.train(train, method='majority', stopcrit='number of iterations', niter=50, track=True, display=False)
model_majority = misc.align(truemodel, model_majority)

model_majority.predict(train, subset=range(model_majority.m), label=label)
print model_majority.log_likelihood(train)

# disp.show_rec(foo)
# disp.comp_heatmap(SEM.C)
#
# disp.comp_heatmap(truemodel.C, list=SEM.activeset)
# disp.comp_heatmap(SEM.C, list=SEM.activeset)

plt.show()

cPickle.dump(rec, open('result.p', 'wb'))

print 1

# mat = misc.contribution(SEM.W, SEM.C)
#
# print 1
#
# ###################### run as script
#
# folder = "/home/vincent/Dropbox/Research/Project/Stryker_Mouse/"
# file = "model_151211data007"
# modeleg = MDPD.StageEM()
# modeleg.load(folder + file + "_eg")
# modelegr = MDPD.StageEM()
# modelegr.load(folder + file + "_egr")
# modelegs = MDPD.StageEM()
# modelegs.load(folder + file+'_egs')
#
# outeg = modeleg.predict()
# modeleg.label = modeleg.label.squeeze()
# outegr = modelegr.predict()
# modelegr.label = modelegr.label.squeeze()
# outegs = modelegs.predict()
# modelegs.label = modelegs.label.squeeze()
#
# ori_yeg = []
# ori_yegr = []
# ori_yegs = []
#
# plt.figure(1)
# for i in range(6):
#     f = plt.subplot(3,2,i+1)
#     f.set_xlim((1,12))
#     plt.hist(modelegr.label[outegr == i], bins=12)
#
# plt.show()
#
#
#
# for i in range(6):
#     foo = np.histogram(modeleg.label[outeg == i], bins=12)
#     ori_yeg.append(foo[0]/float(modeleg.label.size))
#     foo = np.histogram(modelegr.label[outegr == i], bins=12)
#     ori_yegr.append(foo[0]/float(modelegr.label.size))
#     foo = np.histogram(modelegs.label[outegs == i], bins=12)
#     ori_yegs.append(foo[0]/float(modelegs.label.size))
#
# ori_yeg = np.asarray(ori_yeg)
# ori_yegr = np.asarray(ori_yegr)
# ori_yegs = np.asarray(ori_yegs)
#
# cont_eg = misc.contribution(modeleg.W, modeleg.C)
# cont_egr = misc.contribution(modelegr.W, modelegr.C)
# cont_egs = misc.contribution(modelegs.W, modelegs.C)
#
# plt.figure(2)
# for i in range(41):
#     plt.subplot(6,7,i+1)
#     foo = cont_eg[i][0, :]
#     bar = np.dot(foo, ori_yeg)
#     plt.plot(bar)
#     foo = cont_egr[i][0, :]
#     bar = np.dot(foo, ori_yegr)
#     plt.plot(bar)
#     foo = cont_egs[i][0, :]
#     bar = np.dot(foo, ori_yegs)
#     plt.plot(bar)
# plt.show()
