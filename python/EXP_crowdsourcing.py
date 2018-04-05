from __future__ import division

import os
import numpy as np
import time, timeit
import signal
import scipy.io as scio
from scipy.sparse import coo_matrix
from MDPD import *
from MDPD.readers import *
import matplotlib.pyplot as plt
import matplotlib

def handle(signum, frame):
    raise Exception('end of time')

##### analytic tools
def missing_heatmap(train):
    heatmap = train.sum(axis=2)
    y, x = heatmap.shape
    plt.figure()
    plt.pcolor(heatmap, cmap=matplotlib.cm.Blues)
    plt.xlim([0, x])
    plt.ylim([0, y])
    plt.show()
    plt.xlabel('worker number',fontsize=16)
    plt.ylabel('item number', fontsize=16)

def fsmv_curve(feature_rank, train, test, label, f_range, ncomp, niter=50):
    """
    Accuracy curve for feature selection majority vote (scan # of featuers)
    """
    mv_err = []
    mvem_err = []
    nquestion = []
    for topN in f_range:
        # feature selection
        features = feature_rank[:topN]
        print len(features)
        train_selected = train[:, features, :]
        valid_sample = train_selected.sum(axis=(1, 2)) > 0
        train_selected = train_selected[valid_sample, :]
        test_selected = test[:, features, :]
        valid_sample = test_selected.sum(axis=(1,2)) > 0
        nquestion.append(sum(valid_sample == True))
        test_selected = test_selected[valid_sample, :]
        label_selected = label[valid_sample]
        # majority vote
        model = MDPD.MDPD()
        model.fit(train_selected, ncomp=ncomp, init='majority', epoch=niter)
        acc = model.accuracy(test_selected, label_selected)
        _, err_mv = mv_predictor(test_selected, label_selected)
        acc *= len(label_selected)
        acc += (test.shape[0] - nquestion[-1]) * (test.shape[2] - 1) / test.shape[2]
        acc /= len(label)
        err_mv *= len(label_selected)
        err_mv += (test.shape[0] - nquestion[-1]) * (test.shape[2] - 1) / test.shape[2]
        err_mv /= len(label)
        mvem_err.append(1 - acc)
        mv_err.append(err_mv)
    return mv_err, mvem_err, nquestion

def fsspec_curve(feature_rank, train, test, label, f_range, ncomp, niter=50, epoch=20):
    signal.signal(signal.SIGALRM, handle)
    specem_err = []
    nquestion = []
    for nfeatures in f_range:
        # feature selection
        features = feature_rank[:nfeatures]
        print len(features)
        train_selected = train[:, features, :]
        valid_sample = train_selected.sum(axis=(1, 2)) > 0
        train_selected = train_selected[valid_sample, :]
        test_selected = test[:, features, :]
        valid_sample = test_selected.sum(axis=(1,2)) > 0
        nquestion.append(sum(valid_sample == True))
        test_selected = test_selected[valid_sample, :]
        label_selected = label[valid_sample]
        # spectral em
        foo = []
        model = MDPD.MDPD()
        for _ in xrange(epoch):
            try:
                signal.alarm(90)
                model.fit(train_selected, ncomp, init='spectral', epoch=niter)
                model.align(test_selected, label_selected, range(model.dim))
                acc = model.accuracy(test_selected, label_selected)
                acc *= len(label_selected)
                acc += (test.shape[0] - nquestion[-1]) * (test.shape[2] - 1) / test.shape[2]
                acc /= len(label)
                foo.append(1 - acc)
            except:
                pass
            signal.alarm(0)
        specem_err.append(foo)
    return specem_err, nquestion

def mv_predictor(train, label):
    votes = train.sum(axis=1).T
    rank = np.argsort(votes, axis=0)[::-1, :]
    if train.shape[0] != len(label):
        return rank[0, :]
    else:
        err = 0
        tie = 0
        for i in xrange(train.shape[0]):
            idx = 0
            while idx < rank.shape[0] - 1 and votes[rank[idx + 1, i], i] == votes[rank[0, i], i]:
                idx += 1
            if label[i] not in rank[:idx + 1, i]:
                err += 1
            else:
                err += idx / (idx + 1)
                if idx:
                     tie += 1
        print 'mv error rate is ' + str(err / len(label))
        print str(tie) + ' ties'
        return rank[0, :], err / len(label)

def indi_rank(test, label):
    # rank features based on their own performance
    ans = np.argsort(test, axis=2)[:, :, ::-1]
    error = [0] * test.shape[1]
    for i in xrange(test.shape[1]):
        for j in xrange(test.shape[0]):
            foo = ans[j, i, :]
            k = 0
            while k < test.shape[2]-1 and test[j, i, foo[k+1]] == test[j, i, foo[k]]:
                k += 1
            foo = foo[:(k+1)].tolist()
            if label[j] not in foo:
                error[i] += 1
            else:
                error[i] += k / (k + 1)
    return np.argsort(error), sorted(error)

folder = '/media/vzhao/Data/crowdsourcing_datasets/'
# folder = '/Users/vincent/Documents/Research/MDPD/crowdsourcing_datasets'
## Bird data
# reader = Crowd_Sourcing_Readers(os.path.join(folder, 'bird', 'bluebird_crowd.txt'), os.path.join(folder, 'bird', 'bluebird_truth.txt'))
# train, label = reader.data, reader.labels
# lock = np.zeros(train.shape[1:], dtype=np.bool)

## Dog Data
reader = Crowd_Sourcing_Readers(os.path.join(folder, 'dog', 'dog_crowd.txt'), os.path.join(folder, 'dog', 'dog_truth.txt'))
train, label = reader.data, reader.labels
lock = np.zeros(train.shape[1:],dtype=np.bool)
lock[:, -1] = 1


## Web Data
# train = Crowd_Sourcing_Readers.read_data(os.path.join(folder, 'web', 'web_crowd.txt'))
# label = Crowd_Sourcing_Readers.read_label(os.path.join(folder, 'web', 'web_truth.txt'))
# lock = np.zeros(train.shape[1:],dtype=np.bool)
# lock[:, -1] = 1

## TREC
# train = Crowd_Sourcing_Readers.read_data(os.path.join(folder, 'trec', 'trec_crowd.txt'))
# label = Crowd_Sourcing_Readers.read_label(os.path.join(folder, 'trec', 'trec_truth.txt'))
# lock = np.zeros(train.shape[1:],dtype=np.bool)
# lock[:, -1] = 1

# analysys
score_origin = utils.Feature_Selection.MI_score(train, rm_diag=True, lock=lock)

features, score = utils.Feature_Selection.MI_feature_ranking(train, lock=lock)
Ntop = 40
model = MDPD.MDPD()
model.fit(train, ncomp=5, init='majority', verbose=False, features=features[:Ntop], epoch=50, lock=lock)
model.accuracy(train, label)
model.MI_residue(train)







########## to generate ICML figure wrapper
score = MDPD.utils.Feature_Selection.MI_score(train, rm_diag=True, lock=lock)
print np.sum(score)

model = MDPD.MDPD()
model.fit(train, ncomp=4, verbose=False, lock=lock)
model.accuracy(train, label)

logpost = model.log_posterior(train)
score, weights = MDPD.utils.Feature_Selection.MI_score_conditional(train, logpost, rm_diag=True, lock=lock)
print np.sum(score.sum(axis=(0, 1)) * weights)

features, score = MDPD.utils.Feature_Selection.MI_feature_ranking(train, lock=lock)
model.fit(train, ncomp=4, verbose=False, features=features[:60], lock=lock)
model.accuracy(train, label)

logpost = model.log_posterior(train)
score, weights = MDPD.utils.Feature_Selection.MI_score_conditional(train, logpost, rm_diag=True, lock=lock)
print np.sum(score.sum(axis=(0, 1)) * weights)



# feature_rank, vals = model.init_topNfeatures(train_pad, model.dim)

plt.figure(figsize=(8, 2))
plt.plot(score)
# plt.title('Mutual Information Score (eq 1)', fontsize=20)
plt.ylabel('Score', fontsize=16)

x_range = range(1, 40)
feature_rank, _ = MDPD.utils.MI_feature_ranking(train)
mv_err, mvem_err, nq = fsmv_curve(feature_rank, train, train, label, x_range, 2)



x_range = range(5, 40)
spec_err, _ = fsspec_curve(feature_rank, train, train, label,
                           x_range, 2, epoch=2)

med_spec = np.median(spec_err, axis=1)
med_low = np.percentile(spec_err, 25, axis=1)
med_hi = np.percentile(spec_err, 75, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(x_range, med_spec)
plt.fill_between(x_range, med_low, med_hi, facecolor='blue', alpha=0.2, interpolate=True)

x_range = range(1, 40)
plt.plot(x_range, mv_err)
plt.plot(x_range, mvem_err)

feature_rank, _ = indi_rank(train, label)
x_range = range(1, 40)
mv_err, mvem_err, _ = fsmv_curve(feature_rank, train, train, label, x_range, 2)
plt.plot(x_range, mv_err, '--')
plt.plot(x_range, mvem_err, '--')

plt.title('Bird Dataset', fontsize=20)
plt.ylabel('mis-clustering rate (%)', fontsize=16)
plt.xlabel('# of features (workers)', fontsize=16)
plt.legend(['opt-D&S', 'MV', 'MV+EM', r'MV*' , r'MV+EM*'])


########## Dog
# file = '/media/vincent/Data/Dataset/dog/data.mat'
# mat = scio.loadmat(file)
# label = np.asarray(mat['label']).squeeze()-1
# train = mat['z']
# train = np.moveaxis(train, [0, 1, 2], [2, 0, 1])
# train = train[:, :, :]
# train_pad = np.append(train, (1 - train.sum(axis=2))[:, :, np.newaxis], axis=2)

folder = '/home/vincent/Documents/Research/MDPD/crowdsourcing_datasets/dog'
train = readers.read_data(os.path.join(folder, 'dog_crowd.txt'))
label = readers.read_label(os.path.join(folder, 'dog_truth.txt'))
train_pad = np.append(train, (1 - train.sum(axis=2))[:, :, np.newaxis], axis=2)

########## to generate ICML figure wrapper
model = MDPD.MDPD(train_pad, 4)
feature_rank, _ = MDPD.utils.MI_feature_selection(train, 15)


feature_rank, vals = model.init_topNfeatures(train_pad, model.dim, remove_last=False)

plt.figure(figsize=(8, 2))
plt.plot(vals)
plt.ylabel('Score', fontsize=16)

x_range = range(1, 110)
mv_err_algo, mvem_err_algo, _ = fsmv_curve(feature_rank, train, train, label, x_range, 4)

x_range = range(10, 110)
spec_err, _ = fsspec_curve(feature_rank, train, train, label, x_range, 4, epoch=20)
med_spec = np.median(spec_err, axis=1)
med_low = np.percentile(spec_err, 25, axis=1)
med_hi = np.percentile(spec_err, 75, axis=1)

feature_rank, _ = indi_rank(train, label)
mv_err_bench, mvem_err_bench, nq = fsmv_curve(feature_rank, train, train, label, x_range, 4)

plt.figure(figsize=[8,5])
plt.plot(x_range, med_spec)
plt.fill_between(x_range, med_low, med_hi, facecolor='blue', alpha=0.2, interpolate=True)
plt.plot(x_range, mv_err_algo)
plt.plot(x_range, mvem_err_algo)
plt.plot(x_range, mv_err_bench, '--')
plt.plot(x_range, mvem_err_bench, '--')

plt.title('Dog Dataset',fontsize=20)
plt.ylabel('mis-clustering rate (%)',fontsize=16)
plt.ylim([0.14, 0.22])
plt.xlabel('# of features (workers)',fontsize=16)
plt.legend(['opt-D&S', 'MV', 'MV+EM', 'MV*', 'MV+EM*'])


## figure 2
model = MDPD.MDPD(train_pad, 4)
feature_rank1 = model.init_topNfeatures(train_pad, model.dim)
feature_rank2 = indi_rank(train, label)

plt.figure(figsize=(8,5))
x_range, mv_err, mvem_err, nq = fsmv_curve(feature_rank1, train, train, label, 1, 109, 4)
plt.plot(x_range, mv_err)
plt.plot(x_range, mvem_err)
x_range, mv_err, mvem_err, nq = fsmv_curve(feature_rank2, train, train, label, 1, 109, 4)
plt.plot(x_range, mv_err, '--')
plt.plot(x_range, mvem_err, '--')
plt.title('Dog Dataset')
plt.ylabel('mis-clustering rate (%)')
plt.ylim([0.14, 0.22])
plt.xlabel('# of features (workers)')
plt.legend(['MV', 'MV+EM', 'MV*', 'MV+EM*)'])

# random permute labels
# for i in range(dim):
#     idx = np.random.permutation(4)
#     idx = np.append(idx, 4)
#     data[:, i, :] = data[:, i, idx]
# print idx


# stagewise EM
model = MDPD.MDPD(train_pad, 4)
model.reset(train_pad)
model.lock[:, 4] = 0
model.fit(train_pad, ncomp=4, init="StageEM", epoch=50, reset=False)
model.align(train_pad, label, range(model.dim))
model.predict(train_pad, label, subset=range(model.dim))
model.refine(train_pad)
model.predict(train_pad, label, subset=range(model.dim))

# spec
model = MDPD.MDPD(train, 4)
model.fit(train, init="spectral", epoch=2)
model.predict(train, label, subset=range(model.dim))

# spec with infoset
model = MDPD.MDPD(train_pad, 4)
infoset = model.init_topNfeatures(train_pad, 40)
infoset_size = len(infoset)
print len(infoset)
valid_sample = train[:, infoset, :].sum(axis=1).sum(axis=1) > 0
train_info = train[:, infoset, :]
train_info = train_info[valid_sample, :, :]
label_info = label[valid_sample]

# random infoset
infoset = np.random.permutation(model.dim)[:infoset_size]
print len(infoset)
train_info = train[:, infoset, :]

model = MDPD.MDPD(train_info, 4)
model.fit(train_info, init='spectral', epoch=10)
model.predict(train_info, label_info, subset=range(model.dim))

# majority vote
model = MDPD.MDPD(train, 4)
model.fit(train, init='majority', epoch=30)
model.predict(train, label, subset=range(model.dim))
_ = mv_predictor(train, label)
print sum(train.sum(axis=1).argmax(axis=1) - label!=0) / len(label)

# majority vote with infoset
model = MDPD.MDPD(train_info, 4)
model.fit(train_info, init='majority', epoch=30)
model.predict(train_info, label_info, subset=range(model.dim))
_ = mv_predictor(train_info, label_info)
print sum(train_info.sum(axis=1).argmax(axis=1) - label_info!=0) / len(label)


# # Artificial
# nsample = 1000
# dim = 100
# ncomp = 4
# rang = [0.25, 0.27]
# Wgen, Cgen = model_gen.crowdsourcing_rand(dim, ncomp, rang)
# Wgen = np.array(Wgen)
# Cgen = np.array(Cgen)
#
# model = MDPD.MDPD()
# model.get_config(dim=dim, nsample=nsample, ncomp=ncomp, nvocab=ncomp)
# model.get_modelgen(Wgen, Cgen)
# data, label = model.gen_Data(nsample)
# data = np.array(data).swapaxes(0,2).swapaxes(1,2)
#
### RTE
train = np.zeros([800, 164, 2])
with open('/media/vincent/Data/Dataset/CrowdSourcing/rte/rte_crowd.txt', 'r') as h:
    for line in h:
        j, i, l = line[:-1].split('\t')
        j, i, l = int(j)-1, int(i)-1, int(l)-1
        train[j, i, l] = 1

train_pad = np.append(train, (1 - train.sum(axis=2))[:, :, np.newaxis], axis=2)
label = np.zeros(800, dtype=np.int)
with open('/media/vincent/Data/Dataset/CrowdSourcing/rte/rte_truth.txt') as h:
    for line in h:
        j, l = line[:-1].split('\t')
        j, l = int(j)-1, int(l)-1
        label[j] = l

########## to generate ICML figure wrapper
model = MDPD.MDPD(train_pad, 2)
feature_rank, vals = model.init_topNfeatures(train_pad, model.dim, remove_last=True)

plt.figure(figsize=(8, 2))
plt.plot(vals)
plt.ylabel('Score', fontsize=16)

x_range = range(1, 165)
mv_err_algo, mvem_err_algo, _ = fsmv_curve(feature_rank, train, train, label, x_range, 2)

x_range = range(10, 165)
spec_err, _ = fsspec_curve(feature_rank, train, train, label, x_range, 2, epoch=20)
med_spec = np.median(spec_err, axis=1)
med_low = np.percentile(spec_err, 25, axis=1)
med_hi = np.percentile(spec_err, 75, axis=1)

feature_rank, _ = indi_rank(train, label)
mv_err_bench, mvem_err_bench, nq = fsmv_curve(feature_rank, train, train, label, x_range, 2)


plt.figure(figsize=(8,5))
plt.plot(x_range, med_spec)
plt.fill_between(x_range, med_low, med_hi, facecolor='blue', alpha=0.2, interpolate=True)
plt.plot(x_range, mv_err_algo)
plt.plot(x_range, mvem_err_algo)
plt.plot(x_range, mv_err_bench, '--')
plt.plot(x_range, mvem_err_bench, '--')

plt.title('RTE Dataset', fontsize=20)
plt.ylabel('mis-clustering rate (%)', fontsize=16)
plt.ylim([0.06, 0.2])
plt.xlabel('# of features (workers)', fontsize=16)
plt.legend(['opt-D&S', 'MV', 'MV+EM', r'MV*' , r'MV+EM*'])

## figure 2
model = MDPD.MDPD(train_pad, 2)
feature_rank1 = model.init_topNfeatures(train_pad, model.dim)
feature_rank2 = indi_rank(train, label)

plt.figure(figsize=(8,5))
x_range, mv_err, mvem_err, nq = fsmv_curve(feature_rank1, train, train, label, 1, 164, 2)
plt.plot(x_range, mv_err)
plt.plot(x_range, mvem_err)
x_range, mv_err, mvem_err, nq = fsmv_curve(feature_rank2, train, train, label, 1, 164, 2)
plt.plot(x_range, mv_err, '--')
plt.plot(x_range, mvem_err, '--')
plt.title('RTE Dataset')
plt.ylabel('mis-clustering rate (%)')
plt.ylim([0.06, 0.2])
plt.xlabel('# of features (workers)')
plt.legend(['MV', 'MV+EM', 'MV(cheat)', 'MV+EM(cheat)'])


# stagewise EM
model = MDPD.MDPD(train_pad, 2)
model.reset(train_pad)
model.lock[:, 2] = 0
model.fit(train_pad, ncomp=2, init="StageEM", epoch=50, reset=False, feature_init=100)
model.align(train_pad, label, range(model.dim))
model.predict(train_pad, label, subset=range(model.dim))
model.refine(train_pad)
model.predict(train_pad, label, subset=range(model.dim))

# spec
foo = []
for _ in xrange(20):
    model = MDPD.MDPD(train, 2)
    model.fit(train, init="spectral", epoch=2)
    _, err = model.predict(train, label, subset=range(model.dim))
    foo.append(err)

# spec with infoset
model = MDPD.MDPD(train_pad, 2)
infoset = model.init_topNfeatures(train_pad, 20)
infoset_size = len(infoset)
print len(infoset)
valid_sample = train[:, infoset, :].sum(axis=1).sum(axis=1) > 0
train_info = train[:, infoset, :]
train_info = train_info[valid_sample, :, :]
label_info = label[valid_sample]

# random infoset
infoset = np.random.permutation(model.dim)[:infoset_size]
print len(infoset)
train_info = train[:, infoset, :]

model = MDPD.MDPD(train_info, 2)
model.fit(train_info, init='spectral', epoch=10)
model.align(train_info, label, range(model.dim))
model.predict(train_info, label, subset=range(model.dim))

# majority vote
model = MDPD.MDPD(train, 2)
model.fit(train, init='majority', epoch=20)
model.predict(train, label, subset=range(model.dim))
_ = mv_predictor(train, label)
print sum(train.sum(axis=1).argmax(axis=1) - label!=0) / len(label)

# majority vote with infoset
model = MDPD.MDPD(train_info, 2)
model.fit(train_info, init='majority', epoch=20)
model.predict(train_info, label, subset=range(model.dim))
_ = mv_predictor(train_info, label_info)
print sum(train_info.sum(axis=1).argmax(axis=1) - label_info!=0) / len(label)


### Trec
train = np.zeros([19033, 762, 2])
with open('/media/vincent/Data/Dataset/CrowdSourcing/trec/trec_crowd.txt', 'r') as h:
    for line in h:
        j, i, l = line[:-1].split('\t')
        j, i, l = int(j)-1, int(i)-1, int(l)-1
        train[j, i, l] = 1

train_pad = np.append(train, (1 - train.sum(axis=2))[:, :, np.newaxis], axis=2)
foo = []
label = []
with open('/media/vincent/Data/Dataset/CrowdSourcing/trec/trec_truth.txt', 'r') as h:
    for line in h:
        j, l = line[:-1].split('\t')
        j, l = int(j)-1, int(l)-1
        label.append(l)
        foo.append(j)

test = train[foo, :, :]
label = np.array(label)

########## to generate ICML figure wrapper
model = MDPD.MDPD(train_pad, 2)
feature_rank, vals = model.init_topNfeatures(train_pad, model.dim, remove_last=False)
plt.figure(figsize=(8, 2))
plt.plot(vals)
plt.ylabel('Score', fontsize=16)

x_range = range(30, 762, 5)
spec_err, _ = fsspec_curve(feature_rank, train, test, label, x_range, 2, epoch=20, niter=10)

med_spec = np.median(spec_err, axis=1)
med_low = np.percentile(spec_err, 25, axis=1)
med_hi = np.percentile(spec_err, 75, axis=1)

plt.figure(figsize=(8,5))
plt.plot(x_range, med_spec)
plt.fill_between(x_range, med_low, med_hi, facecolor='blue', alpha=0.2, interpolate=True)

x_range = range(1, 762)
mv_err_algo, mvem_err_algo, nq = fsmv_curve(feature_rank, train, test, label, x_range, 2)
plt.plot(x_range, mv_err_algo)
plt.plot(x_range, mvem_err_algo)

feature_rank, errs = indi_rank(test, label)
mv_err_bench, mvem_err_bench, nq = fsmv_curve(feature_rank, train, test, label, x_range, 2)
plt.plot(x_range, mv_err_bench, '--')
plt.plot(x_range, mvem_err_bench, '--')

plt.title('TREC Dataset', fontsize=20)
plt.ylabel('mis-clustering rate (%)', fontsize=16)
plt.xlabel('# of features (workers)', fontsize=16)
plt.legend(['opt-D&S', 'MV', 'MV+EM', 'MV*', 'MV+EM*'])


# spec
foo = []

model = MDPD.MDPD(train, 2)
model.fit(train, init="spectral", epoch=30)
model.align(test, label, range(model.dim))
_, err = model.predict(test, label, subset=range(model.dim))
foo.append(err)

# spec with infoset
model = MDPD.MDPD(train_pad, 2)
infoset = model.init_topNfeatures(train_pad, 300)
print len(infoset)

train_info = train[:, infoset, :]
valid_sample = train_info.sum(axis=(1,2)) > 0
train_info = train_info[valid_sample, :]
test_info = test[:, infoset, :]
valid_sample = test_info.sum(axis=(1,2)) > 0
test_info = test_info[valid_sample, :, :]
label_info = label[valid_sample]
print len(label), len(label_info)

model = MDPD.MDPD(train_info, 2)
model.fit(train_info, init='spectral', epoch=10)
model.predict(test_info, label_info, subset=range(model.dim))

# majority vote
model = MDPD.MDPD(train, 2)
model.fit(train, init='majority', epoch=20)
model.predict(test, label, subset=range(model.dim))
_ = mv_predictor(test, label)

# majority vote with infoset
model = MDPD.MDPD(train_info, 2)
model.fit(train_info, init='majority', epoch=20)
model.predict(test_info, label_info, subset=range(model.dim))
_ = mv_predictor(test_info, label_info)

# fs curve
mvem_err, mv_err, nq = fsmv_curve(train, train_pad, label, 1, 762, 2, test=test)
plt.figure;plt.plot(range(1,763), mvem_err);plt.plot(range(1,763), mv_err)
plt.title('TREC Dataset')
plt.ylabel('mis-clustering rate (%)')
plt.xlabel('# of features (workers)')
plt.legend(['MV+EM', 'MV'])


### Web
train = np.zeros([2665, 177, 5])
with open('/media/vincent/Data/Dataset/CrowdSourcing/web/Train_Query_Doc_ML.TXT', 'r') as h:
    for line in h:
        j, i, l = line[:-1].split('\t')
        j, i, l = int(j)-1, int(i)-1, int(l)-1
        train[j, i, l] = 1

train_pad = np.append(train, (1 - train.sum(axis=2))[:, :, np.newaxis], axis=2)
label = np.zeros(2665, dtype=np.int)
with open('/media/vincent/Data/Dataset/CrowdSourcing/web/Test_Query_Doc_ML.TXT', 'r') as h:
    for line in h:
        j, l = line[:-1].split('\t')
        j, l = int(j)-1, int(l)-1
        label[j] = l


########## to generate ICML figure wrapper
model = MDPD.MDPD(train_pad, 5)
feature_rank, vals = model.init_topNfeatures(train_pad, model.dim)
plt.figure(figsize=(8, 2))
plt.plot(vals)
plt.ylabel('Score', fontsize=16)

x_range = range(10, 178)
spec_err, _ = fsspec_curve(feature_rank, train, train, label, x_range, 5, epoch=20, niter=100)

med_spec = map(np.median, spec_err)
med_low = map(lambda x: np.percentile(x, 25), spec_err)
med_hi = map(lambda x: np.percentile(x, 75), spec_err)


plt.figure(figsize=(8,5))
plt.plot(x_range, med_spec)
plt.fill_between(x_range, med_low, med_hi, facecolor='blue', alpha=0.2, interpolate=True)

x_range = range(1, 178)
mv_err, mvem_err, _ = fsmv_curve(feature_rank, train, train, label, x_range, 5, niter=100)
plt.plot(x_range, mv_err)
plt.plot(x_range, mvem_err)

feature_rank = indi_rank(train, label)
mv_err, mvem_err, _ = fsmv_curve(feature_rank, train, train, label, x_range, 5, niter=100)
plt.plot(x_range, mv_err, '--')
plt.plot(x_range, mvem_err, '--')

plt.title('Web Dataset', fontsize=20)
plt.ylabel('mis-clustering rate (%)', fontsize=16)
plt.ylim([0.08, 0.3])
plt.xlabel('# of features (workers)', fontsize=16)
plt.legend(['opt-D&S', 'MV', 'MV+EM', 'MV*', 'MV+EM*'])
plt.show(block=False)


## figure 2
model = MDPD.MDPD(train_pad, 5)
feature_rank1 = model.init_topNfeatures(train_pad, model.dim)
feature_rank2 = indi_rank(train, label)

plt.figure(figsize=(8,5))
x_range, mv_err, mvem_err, nq = fsmv_curve(feature_rank1, train, train, label, 1, 177, 5, niter=100)
plt.plot(x_range, mv_err)
plt.plot(x_range, mvem_err)
x_range, mv_err, mvem_err, nq = fsmv_curve(feature_rank2, train, train, label, 1, 177, 5, niter=100)
plt.plot(x_range, mv_err, '--')
plt.plot(x_range, mvem_err, '--')
plt.title('Web Dataset')
plt.ylabel('mis-clustering rate (%)')
plt.ylim([0.05, 0.3])
plt.xlabel('# of features (workers)')
plt.legend(['MV', 'MV+EM', 'MV(cheat)', 'MV+EM(cheat)'])

# stagewise EM
model = MDPD.MDPD(train_pad, 5)
model.reset(train_pad)
model.lock[:, 2] = 0
model.fit(train_pad, ncomp=5, init="StageEM", epoch=50, reset=False, feature_init=12)
model.align(train_pad, label, range(model.dim))
model.predict(train_pad, label, subset=range(model.dim))
model.refine(train_pad)
model.predict(train_pad, label, subset=range(model.dim))
infoset = model.infoset

# spec
foo = []
for _ in xrange(20):
    model = MDPD.MDPD(train, 5)
    model.fit(train, init="spectral", epoch=30)
    _, err = model.predict(train, label, subset=range(model.dim))
    foo.append(err)

# spec with infoset
model = MDPD.MDPD(train_pad, 5)
infoset = model.init_topNfeatures(train_pad, 10, remove_last=True)
print len(infoset)
train_info = train[:, infoset, :]
valid_sample = train_info.sum(axis=(1,2)) > 0
train_info = train_info[valid_sample, :]
label_info = label[valid_sample]



# # random infoset
# infoset = np.random.permutation(model.dim)[:infoset_size]
# print len(infoset)
# train_info = train[:, infoset, :]

model = MDPD.MDPD(train_info, 5)
model.fit(train_info, init='spectral', epoch=30)
model.predict(train_info, label_info, subset=range(model.dim))

# majority vote
model = MDPD.MDPD(train, 5)
# model.lock[:, -1] = 0
model.fit(train, init='majority', epoch=100)
model.predict(train, label, subset=range(model.dim))
mv_predictor(train, label)

# majority vote with infoset
model = MDPD.MDPD(train_info, 5)
model.fit(train_info, init='majority', epoch=100)
model.predict(train_info, label_info, subset=range(model.dim))
mv_predictor(train_info, label_info)