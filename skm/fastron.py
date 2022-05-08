from traceback import print_tb
import numpy as np


class Fastron:
    def __init__(self, input_data, y):
        # gamma (kernel width), beta (conditional bias)
        self.gamma = 30
        self.beta = 1

        # max update iterations, max number of support points
        self.maxUpdates = 1000
        self.maxSupportPoints = 500

        # count of points with nonzero weights
        self.numberSupportPoints = 0

        # dataset
        self.data = input_data

        # support points

        # number of datapoints and dimensionality
        self.N = self.data.shape[0]
        self.dim = self.data.shape[1]

        # Gram matrix
        self.G = np.zeros(shape=(self.N, self.N))

        # weights, score function, true labels
        self.alpha = np.zeros(shape=(self.N))
        self.F = np.zeros(shape=(self.N))
        self.y = y

        self.gramComputed = np.zeros(shape=(self.N))

    def kernel(self, target, type):
        if type == 1:
            number = self.numberPosSupportPoints
            r2 = np.ones(shape=(number))
            for j in range(self.dim):
                r2 += self.gamma/2 * \
                    (self.pos_points[0:number, j:j+1].ravel() - target[j])**2
        else:
            number = self.numberNegSupportPoints
            r2 = np.ones(shape=(number))
            for j in range(self.dim):
                r2 += self.gamma/2 * \
                    (self.neg_points[0:number, j:j+1].ravel() - target[j])**2
        # k = 1 / np.exp(r2)[:, np.newaxis]
        k = 1 / (r2*r2)[:, np.newaxis]
        return k

    def predict(self, datas):
        predicts = np.empty(shape=(len(datas)))
        for i in range(len(datas)):
            target = datas[i]
            f = np.dot(self.pos_alpha, self.kernel(target, 1)) + \
                np.dot(self.neg_alpha, self.kernel(target, 0))
            predicts[i] = (1 if f > 0 else -1)
        return predicts

    def updateModel(self):
        margin = self.y*self.F
        for i in range(self.maxUpdates):
            # print("==========iteration: ", i)
            margin = self.y*self.F
            # print("margin:")
            # print(margin)
            min_ = np.min(margin)
            # print(min_)
            # Margin-based priotization
            if min_ <= 0:
                idx = np.argmin(margin)
                # print("idx: ", idx)
                if self.gramComputed[idx] == 0:
                    # print("in computeGram")
                    self.computeGramMatrixCol(idx)
                # One-step weight correction
                delta = (-1.0 if self.y[idx] < 0 else self.beta) - self.F[idx]
                # print("delta: ", delta)
                if self.alpha[idx] > 0:
                    self.alpha[idx] += delta
                    self.F += self.G[0:self.N, idx:idx+1].ravel()*delta
                    # print("F1")
                    # print(self.F)
                    continue
                elif self.numberSupportPoints < self.maxSupportPoints:
                    self.alpha[idx] = delta
                    self.F += self.G[0:self.N, idx:idx+1].ravel()*delta
                    # print("F2")
                    # print(self.F)
                    self.numberSupportPoints += 1
                    continue

            # Remove redundant points
            max_, idx = self.calculateMarginRemoved(idx)
            if max_ > 0:
                self.F -= self.G[0:self.N, idx:idx+1].ravel()*self.alpha[idx]
                self.alpha[idx] = 0
                margin = self.y*self.F
                self.numberSupportPoints -= 1
                print("removed sp")
                continue

            if self.numberSupportPoints == self.maxSupportPoints:
                print("Fail: Hit support point limit in {} iterations!".format(i))
                self.sparsity()
                return
            else:
                print("Success: Model update complete in {} iterations!".format(i))
                self.sparsity()
                return
        print("Failed to converge after {} iterations!".format(self.maxUpdates))
        self.sparsity()
        return

    def calculateMarginRemoved(self, idx):
        max_ = 0
        for i in range(self.N):
            if self.alpha[i] > 0:
                removed = self.y[i]*(self.F[i]-self.alpha[i])
                if removed > max_:
                    max_ = removed
                    idx = i
        return max_, idx

    def computeGramMatrixCol(self, idx, startIdx=0):
        r2 = np.ones(shape=(self.N-startIdx))
        for j in range(self.dim):
            r2 += self.gamma/2 * \
                (self.data[startIdx:self.N, j:j+1].ravel() -
                 self.data[idx][j])**2
        self.G[startIdx:self.N, idx:idx+1] = 1 / (r2*r2)[:, np.newaxis]
        # self.G[startIdx:self.N, idx:idx+1] = 1 / np.exp(r2)[:, np.newaxis]
        self.gramComputed[idx] = 1
        # print(self.G.round(6))

    def sparsity(self):
        keep = np.nonzero(self.alpha)
        keep_inds = keep[0]
        self.numberSupportPoints = keep_inds.size

        self.data = self.data[keep_inds]
        label = self.y[keep_inds]

        self.pos_points = self.data[np.nonzero(label+1)]
        self.neg_points = self.data[np.nonzero(label-1)]

        self.alpha = self.alpha[keep_inds]
        self.pos_alpha = self.alpha[np.nonzero(label+1)]
        self.neg_alpha = self.alpha[np.nonzero(label-1)]

        self.numberPosSupportPoints = self.pos_alpha.size
        self.numberNegSupportPoints = self.neg_alpha.size

        self.gramComputed = self.gramComputed[keep_inds]
        self.F = self.F[keep_inds]
        self.y = self.y[keep_inds]

        temp = np.zeros(shape=(self.numberSupportPoints,
                        self.numberSupportPoints))
        for idx_i, val_i in enumerate(keep_inds):
            for idx_j, val_j in enumerate(keep_inds):
                temp[idx_i][idx_j] = self.G[val_i][val_j]
        self.G = temp
