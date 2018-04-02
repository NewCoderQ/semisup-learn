# -*- coding: utf-8 -*-
"""
    An example to exculate the Semi-supervised learning
"""

import sys
sys.path.append('../')

import numpy as np
import random
from frameworks.CPLELearning import CPLELearningModel
from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import sklearn.svm
from methods.scikitWQDA import WQDA
from frameworks.SelfLearning import SelfLearningModel

# load data
# fetch_mldata
# cancer = fetch_mldata("Lung cancer (Ontario)")        # load data
cancer = fetch_mldata("heart")        # load data

X = cancer.target.T     # label (270, )
ytrue = np.copy(cancer.data).flatten()      # data (3510, )
ytrue[ytrue>0]=1        # set the value = 1 where the original value > 0

# label a few points 
labeled_N = 4
ys = np.array([-1]*len(ytrue)) # -1 denotes unlabeled point

# print(list(np.where(ytrue == 0)[0]))

random_labeled_points = random.sample(list(np.where(ytrue == 0)[0]), int(labeled_N/2))+\
                        random.sample(list(np.where(ytrue == 1)[0]), int(labeled_N/2))

# set the labels of the labeled samples
ys[random_labeled_points] = ytrue[random_labeled_points]
2
# print(X[random_labeled_points])
# supervised score 
# basemodel = WQDA() # weighted Quadratic Discriminant Analysis
# SGDClassifier
basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
# model fit
basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
print ("supervised log.reg. score", basemodel.score(X, ytrue))

# self learning framework
# fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)      # defaule use the sample weighting 
ssmodel.fit(X, ys)
print ("self-learning log.reg. score", ssmodel.score(X, ytrue))

# semi-supervised score (base model has to be able to take weighted samples)
ssmodel = CPLELearningModel(basemodel)
ssmodel.fit(X, ys)
print "CPLE semi-supervised log.reg. score", ssmodel.score(X, ytrue)

# semi-supervised score, WQDA model
ssmodel = CPLELearningModel(WQDA(), predict_from_probabilities=True) # weighted Quadratic Discriminant Analysis
ssmodel.fit(X, ys)
print "CPLE semi-supervised WQDA score", ssmodel.score(X, ytrue)

# # semi-supervised score, RBF SVM model
# ssmodel = CPLELearningModel(sklearn.svm.SVC(kernel="rbf", probability=True), predict_from_probabilities=True) # RBF SVM
# ssmodel.fit(X, ys)
# print "CPLE semi-supervised RBF SVM score", ssmodel.score(X, ytrue)
