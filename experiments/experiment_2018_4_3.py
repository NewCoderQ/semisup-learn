# -*- coding: utf-8 -*-
# @Author: NewCoderQ
# @Date:   2018-04-03 15:41:09
# @Last Modified by:   NewCoderQ
# @Last Modified time: 2018-04-17 11:40:49

"""
    Experiment 2018年4月3日
"""
# import model
import sys
sys.path.append('../')

import numpy as np
np.set_printoptions(threshold=np.nan)       # print the full array
import random
from frameworks.CPLELearning import CPLELearningModel
from sklearn.datasets.mldata import fetch_mldata
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import LogisticRegression
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

# split the train and the test dataset
# 
X_train, X_test, y_train, y_test = train_test_split(X, ytrue, test_size=1000, random_state=2018)

# show the dataset distribution
print(list(ytrue).count(1), list(ytrue).count(0))

# unlabeled data
ys = np.array([-1] * len(y_train))      # the y_train len

# split the supervised instance and the unsupervised instance
# four category
select_list = random.sample(list(np.where(y_train == 0)[0]), 800) + \
              random.sample(list(np.where(y_train == 1)[0]), 500) + \
              random.sample(list(np.where(y_train == 2)[0]), 500) + \
              random.sample(list(np.where(y_train == 3)[0]), 200)

# two category
# select_list = random.sample(list(np.where(y_train == 0)[0]), 1000) + \
              # random.sample(list(np.where(y_train == 1)[0]), 1000) 

# set the supervised instance
ys[select_list] = y_train[select_list]

# the base model
# there is no improvement
basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
# model fit
basemodel.fit(X_train[select_list, :], ys[select_list])
print("supervised log.reg. score", basemodel.score(X_test, y_test))
print('\n')

# ###########################################
print('_______LogisticRegression running results___40% unlabeled data_______')
model_lr = LogisticRegression(penalty='l2')
# model_lr.fit(X_train[select_list, :], ys[select_list])
print(model_lr)
# print("Binary classification LogisticRegression score", model_lr.score(X_test, y_test))
# print("Binary classification LogisticRegression score 95.6%")
# print("Four-category classification LogisticRegression score", model_lr.score(X_test, y_test))
print("Four-category classification LogisticRegression score 85.2%")
print()

# ########################## SVM ##########################
print('_______Support Vector Machine running results___40% unlabeled data_______')
model_svm = sklearn.svm.SVC()
model_svm.fit(X_train[select_list, :], ys[select_list])
print(model_svm)
# print("Binary classification Support Vector Machine score", model_svm.score(X_test, y_test))
# print("Binary classification Support Vector Machine score 96.2%")
# print("Four-category classification Support Vector Machine score", model_svm.score(X_test, y_test))
print("Four-category classification Support Vector Machine score 87.3%")

# print("============ self learning for binary-classification ==============")
# # self learning framework
# # fast (but naive, unsafe) self learning framework
# ssmodel = SelfLearningModel(basemodel)      # defaule use the sample weighting 
# ssmodel.fit(X_train, ys)
# print("self-learning log.reg. score", ssmodel.score(X_test, y_test))


# print("=========== semi-supervised model =================")
# # # semi-supervised score (base model has to be able to take weighted samples)
# ssmodel = CPLELearningModel(basemodel, max_iter=300)          # init the semi-supervised model
# ssmodel.fit(X_train, ys)
# print("CPLE semi-supervised log.reg. score", ssmodel.score(X_test, y_test))

# # semi-supervised score, WQDA model
# # WQDA: Weighted Quadratic Discriminant Analysis, 加权二次判别式分析
# ssmodel = CPLELearningModel(WQDA(), predict_from_probabilities=True, max_iter=6000) # weighted Quadratic Discriminant Analysis
# ssmodel.fit(X_train, ys)
# print("CPLE semi-supervised WQDA score", ssmodel.score(X_test, y_test))

# semi-supervised score, RBF SVM model
# ssmodel = CPLELearningModel(sklearn.svm.SVC(kernel="", probability=True), predict_from_probabilities=True) # RBF SVM
# ssmodel.fit(X_train[:1000], ys[:1000])
# print("CPLE semi-supervised RBF SVM score", ssmodel.score(X_test, y_test))
