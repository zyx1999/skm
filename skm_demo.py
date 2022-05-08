# from skm.skm import SKM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
#
# LOAD PARAMS
#
def load_parameters():
    parameters = {'intel': \
                      ('./data/intel.csv',
                       (0.2, 0.2), # grid resolution for occupied samples and free samples, respectively
                       (-20, 20, -25, 10),  # area to be mapped [x1_min, x1_max, x2_min, x2_max]
                       1, # skip
                       6.71,  # gamma: kernel parameter
                       25, # k_nearest: picking K nearest relevance vectors
                       20 # max_iter: maximum number of iterations
                       ),
                  }
    return parameters['intel']
fn_train, res, cell_max_min, skip, gamma, k_nearest, max_iter = load_parameters()

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

#
# Plot the dataset accumulated from all time steps.
#
# plt.figure()
# plt.scatter(X_train[:, 1], X_train[:, 2], c=Y_train, s=2)
# plt.title('Training data')
# plt.colorbar()
# plt.show()

#
# query locations for plotting map of the environment
#
q_resolution = 0.25
xx, yy = np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
                     np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

# skm = SKM()

max_t = 1
print("Total number of scans = ", max_t)
for ith_scan in range(0, max_t, skip):

    # extract data points of the ith scan
    ith_scan_indx = X_train[:, 0] == ith_scan
    print('{}th scan:\n  N={}'.format(ith_scan, np.sum(ith_scan_indx)))
    y = Y_train[ith_scan_indx]
    X = X_train[ith_scan_indx, 1:]
    print(X.shape)
    clf = SVC(kernel='rbf', gamma=0.6)
    clf.fit(X, y)
    print(clf.support_vectors_.shape)
    # pa = clf.get_params()
    # sa = clf.decision_function(X)
    # print(pa)
    # print(sa.shape)
    h=0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")
    plt.title("3-Class classification using Support Vector Machine with custom kernel")
    plt.axis("tight")
    plt.show()

    # plot data points, relevance vectors and our probabilistic map.
    # plt.figure(figsize=(20, 5))
    # plt.subplot(131)
    # ones_ = np.where(y > 0.5)
    # zeros_ = np.where(y < 0.5)
    # plt.scatter(X[ones_, 0], X[ones_, 1], color=['red'], cmap='jet', s=5, edgecolors=['none'])
    # plt.scatter(X[zeros_, 0], X[zeros_, 1], color=['blue'], cmap='jet', s=5, edgecolors=['none'])

    # plt.title('Data points at t={}'.format(ith_scan))
    # plt.xlim([cell_max_min[0], cell_max_min[1]])
    # plt.ylim([cell_max_min[2], cell_max_min[3]])
    # # plt.subplot(132)
    # # ones_ = np.where(y == 1)

    # plt.savefig('./outputs/imgs/step_' + str(ith_scan).zfill(3) + '.png', bbox_inches='tight')
    # plt.close("all")

