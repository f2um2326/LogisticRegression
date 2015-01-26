# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:40:08 2015

@author: shimba
"""
# print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X = X[y != 2]
y = y[y != 2]

# parameter?
# cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

clf.fit(X, y)

# plot result
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.show()