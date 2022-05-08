import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf

def plot_decision_boundary(X, y, model, ith_scan):
    h = 0.02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.pcolormesh(xx, yy, Z, cmap='jet')
    # Plot also the training points
    ones_ = np.where(y > 0.5)
    zeros_ = np.where(y < 0.5)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.scatter(X[ones_, 0], X[ones_, 1], color=['red'], cmap='jet', s=5, edgecolors=['none'])
    plt.scatter(X[zeros_, 0], X[zeros_, 1], color=['blue'], cmap='jet', s=5, edgecolors=['none'])
    plt.title('Data points at t={}'.format(ith_scan))
    cell_max_min = (x_min, x_max, y_min, y_max)
    plt.xlim([cell_max_min[0], cell_max_min[1]])
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    
    plt.subplot(122)
    pos_points = model.pos_points
    neg_points = model.neg_points
    plt.scatter(pos_points[:, 0], pos_points[:,1], color='red', cmap='jet', s=5, edgecolors=['none'])
    plt.scatter(neg_points[:, 0], neg_points[:,1], color='blue', cmap='jet', s=5, edgecolors=['none'])
    plt.title('support points at t={}'.format(ith_scan))
    plt.xlim([cell_max_min[0], cell_max_min[1]])
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    plt.show()

def plot2(X, y, model):
    line_width = 1 # For plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    cell_max_min = (x_min, x_max, y_min, y_max)
    q_resolution = 0.02
    x = np.arange(cell_max_min[0], cell_max_min[1]+1, q_resolution)
    y = np.arange(cell_max_min[2], cell_max_min[3]+1, q_resolution)
    xx, yy = np.meshgrid(x,y)
    X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
    Y_query = model.predict(X_query)
    plt.xlim([cell_max_min[0], cell_max_min[1]])
    plt.ylim([cell_max_min[2], cell_max_min[3]])
    # plt.contourf(x, y, np.reshape(Y_query, (len(y), len(x))), 5, cmap='Pastel1')
    cs1 = plt.contour(x, y, np.reshape(Y_query, (len(y), len(x))), cmap="tab20c", linestyles="solid", linewidths=line_width)
    # cs2 = plt.contour(x, y, np.reshape(g3, (len(y), len(x))), levels=[0.0], cmap="Greys_r", linestyles="dashed", linewidths=line_width)
    pos_points = model.pos_points
    neg_points = model.neg_points
    plt.scatter(pos_points[:, 0], pos_points[:,1], color='red', cmap='jet', s=5, edgecolors=['none'])
    plt.scatter(neg_points[:, 0], neg_points[:,1], color='blue', cmap='jet', s=5, edgecolors=['none'])
    # plt.title('support points at t={}'.format(ith_scan))
    h1, _ = cs1.legend_elements()
    # h2, _ = cs2.legend_elements()

    plt.legend([h1[0]], ['true boundary'], loc = 2, fontsize=10)
    plt.show()