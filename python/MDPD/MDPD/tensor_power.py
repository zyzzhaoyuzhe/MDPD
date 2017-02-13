"""
Perform Spectral tensor methods
"""

import numpy as np
import sys
from scipy.misc import logsumexp

def tensorpower(m2, m3, k):
    """

    :param m2: 1/n * m2[0] * m2[1]' gives the second order tensor
    :param m3:
    :param k:
    :return: W prior distribution. Cg is the confusion matrix for the group
    """
    n = m2[0].shape[1]
    r = m2[0].shape[0]
    m2_mat = 1./n * np.dot(m2[0], m2[1].transpose())
    u, s, v = np.linalg.svd(m2_mat)

    u = u[:, :k]
    s = s[:k]

    # whitening matrix
    Q = np.dot(u, np.diag(np.sqrt(1./s)))

    # whiten M3
    m3_tilde = []
    for g in range(3):
        m3_tilde.append(np.dot(Q.transpose(), m3[g]))

    #
    eig_val = []
    W = np.zeros(k)
    Cg = np.zeros((r, k))
    eig_vec = np.zeros((r, k))

    #
    tensor = m3_tilde
    for l in range(k):
        s_temp, v_temp = tp_helper(tensor, eig_val, eig_vec, l)
        eig_val.append(s_temp)
        eig_vec[:, l] = v_temp
        W[l] = 1./(s_temp ** 2)
        foo = s_temp * np.dot(np.linalg.pinv(Q.transpose()), v_temp)
        Cg[:, l] = foo/np.sum(foo)

    # align columns
    idx = np.argmax(Cg, axis=0)
    idx = np.argsort(idx)
    if np.unique(idx).size < k:
        pass
    elif np.unique(idx).size == k:
        W = W[idx]
        Cg = Cg[:,idx]
    return W, Cg


def get_tensor(train, a, b, c, ):
    """

    :param train:
    :param a:
    :param b:
    :param c:
    :return: m2 returns matrix needed to generate m2 (1/n*m2[0]*m2[1]'). m3 returns matrix needed to generate the third order tensor.
    """
    # (a, b, c) is the permutation of (1, 2, 3)
    n = train[0].shape[1]
    Acb = 1./n * np.dot(train[c], train[b].transpose())
    Aab = 1./n * np.dot(train[a], train[b].transpose())
    Aca = 1./n * np.dot(train[c], train[a].transpose())
    Aba = 1./n * np.dot(train[b], train[a].transpose())

    #
    Ba = np.dot(np.dot(Acb, np.linalg.inv(Aab)), train[a])
    Bb = np.dot(np.dot(Aca, np.linalg.inv(Aba)), train[b])

    #
    m2 = []
    m2.append(Ba)
    m2.append(Bb)
    m3 = []
    m3.append(Ba)
    m3.append(Bb)
    m3.append(train[c])
    return m2, m3


def get_Ci(train_g, W, Cg, g, data, i):
    n = train_g[0].shape[1]
    foo = 1./n * np.dot(data[:, i, :].T, train_g[g].transpose())
    Ci = np.dot(foo, np.linalg.inv(np.dot(np.diag(W), Cg[g].transpose())))
    # if there is negative entries
    Ci[Ci<0] = 0.01
    # normalize Ci's column
    Ci = Ci/np.sum(Ci, axis=0)
    return Ci


def tp_helper(m3_tilde, eig_val, eig_vec, num, L=10, N=100):
    """
    A helper function to perform robust tensor power method.
    :param m3_tilde: whitened tensor m3
    :param L: iteration number (Tensor decompositions for learning latent variable models)
    :param N: iteration number (Tensor decompositions for learning latent variable models)
    :param eig_val: eigen-values (used to deflat tensor)
    :param eig_vec: eigen-vectors (used to deflat tensor)
    :param num: number of eigen-value/eigen-vector pairs already found (used to deflat tensor
    :return: s and v (a pair of eigen-value and eigen-vector)
    """
    dim = m3_tilde[0].shape[0]
    n = m3_tilde[0].shape[1]

    foo_lambda = np.zeros((L))  # to record eigen-value for each iteration
    foo_vec = np.zeros((dim, L))    # to record eigen-vector for each iteration
    for tau in range(L):
        vec = np.random.random_sample((dim))
        vec = vec/np.linalg.norm(vec)
        for t in range(N):
            temp2 = np.dot(vec, m3_tilde[1])
            temp3 = np.dot(vec, m3_tilde[2])
            temp = temp2 * temp3
            old_vec = vec
            vec = 1./n * np.dot(m3_tilde[0], temp)
            for k in range(num):
                vec = vec - eig_val[k] * eig_vec[:, k] * np.dot(old_vec, eig_vec[:, k])**2
            vec = vec/np.linalg.norm(vec)
        foo_vec[:, tau] = vec
        foo_lambda[tau] = np.mean(np.dot(vec, m3_tilde[0]) *
                                  np.dot(vec, m3_tilde[1]) *
                                  np.dot(vec, m3_tilde[2]))
    idx = np.argmax(foo_lambda)
    v = foo_vec[:, idx]
    s = np.mean(np.dot(v, m3_tilde[0]) *
                np.dot(v, m3_tilde[1]) *
                np.dot(v, m3_tilde[2]))
    return s, v
