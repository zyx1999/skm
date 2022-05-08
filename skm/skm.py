import numpy as np
from pytest import param
from sklearn.svm import SVC
from sklearn.metrics.pairwise import pairwise_kernels
from numpy import linalg as LA

DEFAULT_DECISION_THRESHOLD = 0.1


###############################################################################
#                  Helper Functions
###############################################################################

def gen_kernel(X, Y, gamma, kernel):
    '''
    Calculates kernel features
    '''
    params = {"gamma": gamma}
    return pairwise_kernels(X, Y, metric=kernel, filter_params=True, **params)


