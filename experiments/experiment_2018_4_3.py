# -*- coding: utf-8 -*-
# @Author: NewCoderQ
# @Date:   2018-04-03 15:41:09
# @Last Modified by:   NewCoderQ
# @Last Modified time: 2018-04-04 14:56:54

"""
    Experiment 2018年4月3日
"""
# import model
import sys
sys.path.append('../')

import numpy as np
import random
from frameworks.CPLELearning import CPLELearningModel
from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import sklearn.svm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from methods.scikitWQDA import WQDA
from frameworks.SelfLearning import SelfLearningModel

import dataset

# load data

cataract = dataset.Cataract_data()
X = cataract.data       # data feature
X = preprocessing.scale(X)
ytrue = cataract.label  # data label


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, ytrue, test_size=100, random_state=2018)


print(list(ytrue).count(1), list(ytrue).count(0))
ys = np.array([-1] * len(y_train))

# split the supervised instance and the unsupervised instance
select_list = random.sample(list(np.where(y_train == 0)[0]), 100) + \
              random.sample(list(np.where(y_train == 1)[0]), 100)
              # random.sample(list(np.where(ytrue == 2)[0]), 100) + \
              # random.sample(list(np.where(ytrue == 3)[0]), 100)

# set the supervised instance
ys[select_list] = y_train[select_list]

basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
# model fit
basemodel.fit(X_train[select_list, :], ys[select_list])
print("supervised log.reg. score", basemodel.score(X_test, y_test))

# self learning framework
# fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)      # defaule use the sample weighting 
ssmodel.fit(X_train, ys)
print("self-learning log.reg. score", ssmodel.score(X_test, y_test))

# # semi-supervised score (base model has to be able to take weighted samples)
# ssmodel = CPLELearningModel(basemodel)
# ssmodel.fit(X_train, ys)
# print("CPLE semi-supervised log.reg. score", ssmodel.score(X_test, y_test))

# # semi-supervised score, WQDA model
# # WQDA: Weighted Quadratic Discriminant Analysis, 加权二次判别式分析
# ssmodel = CPLELearningModel(WQDA(), predict_from_probabilities=True, max_iter=6000) # weighted Quadratic Discriminant Analysis
# ssmodel.fit(X_train, ys)
# print("CPLE semi-supervised WQDA score", ssmodel.score(X_test, y_test))

# semi-supervised score, RBF SVM model
ssmodel = CPLELearningModel(sklearn.svm.SVC(kernel="", probability=True), predict_from_probabilities=True) # RBF SVM
ssmodel.fit(X_train[:1000], ys[:1000])
print("CPLE semi-supervised RBF SVM score", ssmodel.score(X_test, y_test))
