from math import floor
from skm.fastron import Fastron
from skm.utils import *
import numpy as np
import pandas as pd
import time

fn_train = './data/intel.csv'
# read data
g = pd.read_csv(fn_train, delimiter=',').values
# 90% for training
X_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 0:3])
Y_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 3][:, np.newaxis]).ravel()  # * 2 - 1
# 10% for testing
X_test = np.float_(g[::10, 0:3])
Y_test = np.float_(g[::10, 3][:, np.newaxis]).ravel()  # * 2 - 1
print(len(g), len(Y_test), len(Y_train))
print(sum(Y_train), sum(Y_test))

tmp=[]
spoint_count = []
max_t = 10
model = Fastron()
print("Total number of scans = ", max_t)
for ith_scan in range(0, max_t, 1):
    # extract data points of the ith scan
    ith_scan_indx = X_train[:, 0] == ith_scan
    print('{}th scan:\n  N={}'.format(ith_scan, np.sum(ith_scan_indx)))
    y = Y_train[ith_scan_indx]
    X = X_train[ith_scan_indx, 1:]
    y[y<0.5] = -1
    start = time.time()
    model.feed(X,y)
    model.updateModel()
    end = time.time()
    spoint_count.append([ith_scan+1, len(model.alpha), len(model.pos_alpha), len(model.neg_alpha)])
    # tmp.append([ith_scan+1, len(y), model.converge_interation, round((end-start)*1000, 2)])
    # debug(model)
    # plot3(X, y, model, ith_scan)
    plot5(X,y,model,ith_scan)
    

# save_params(tmp)
# compare_support_points(spoint_count)

