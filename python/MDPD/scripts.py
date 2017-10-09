import time
import scipy.io as scio
from MDPD import *
import numpy as np
from multiprocessing import Pool
import sys
import matplotlib.pyplot as plt

mvem_err, mv_err, nq = fs_curve(train, train_pad, label, 10, 762, 2, test=test)

plt.figure;plt.plot(mvem_err);plt.plot(mv_err)


plt.figure();plt.plot(nq)