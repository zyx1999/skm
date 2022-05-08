from skm.fastron import Fastron
from skm.utils import plot_decision_boundary, plot2
import numpy as np
import pandas as pd

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


max_t = 1
print("Total number of scans = ", max_t)
for ith_scan in range(0, max_t, 1):
    # extract data points of the ith scan
    ith_scan_indx = X_train[:, 0] == ith_scan
    print('{}th scan:\n  N={}'.format(ith_scan, np.sum(ith_scan_indx)))
    y = Y_train[ith_scan_indx]
    X = X_train[ith_scan_indx, 1:]
    y[y<0.5] = -1
    model = Fastron(X, y)
    model.updateModel()
    # print("amount of support points: ", model.numberSupportPoints)
    # print("alpha: ", model.alpha)
    # print("+ alpha: ", model.pos_alpha)
    # print("- alpha: ", model.neg_alpha)
    # print("dicesion boundary: ", model.F)
    # print("support point: ",model.data)
    # print("+ support point: ",model.pos_points)
    # print("- support point: ",model.neg_points)
    # print(model.G)
    # plot_decision_boundary(X, y, model, ith_scan)
    plot2(X, y, model)
    


