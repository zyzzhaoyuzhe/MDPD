# Jensen Shannon Divergence between two multi-dimensional discrete probability
# prob1[][:,k1] and prob1[][:,k2]
from MDPD.utils import mylog
import numpy as np
import sys


def js(prob1, k1, prob2, k2):
    if not isinstance(prob1, np.ndarray):
        prob1 = np.asarray(prob1)
    if not isinstance(prob2, np.ndarray):
        prob2 = np.asarray(prob2)
    if prob1.shape[0] != prob2.shape[0]:
        sys.stderr('error: Inputs have different dimension')
    else:
        m = prob1.shape[0]

    output = 0
    for i in range(m):
        p1 = prob1[i][:, k1]
        p2 = prob2[i][:, k2]
        p_ave = .5 * (p1 + p2)
        output += 0.5 * (np.dot(p1, mylog(p1 / p_ave)) +
                         np.dot(p2, mylog(p2 / p_ave)))
    return output
