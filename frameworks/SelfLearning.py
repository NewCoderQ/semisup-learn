# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
import sklearn.metrics
import sys
import numpy
from sklearn.linear_model import LogisticRegression as LR

class SelfLearningModel(BaseEstimator):
    """
    Self Learning framework for semi-supervised learning

    This class takes a base model (any scikit learn estimator),
    trains it on the labeled examples, and then iteratively 
    labeles the unlabeled examples with the trained model and then 
    re-trains it using the confidently self-labeled instances 
    (those with above-threshold probability) until convergence.
    
    See e.g. http://pages.cs.wisc.edu/~jerryzhu/pub/sslicml07.pdf

    Parameters
    ----------
    basemodel : BaseEstimator instance can be a classifier from sklearn
        Base model to be iteratively self trained

    max_iter : int, optional (default=200) the max iteration
        Maximum number of iterations

    prob_threshold : float, optional (default=0.8)
        Probability threshold for self-labeled instances
    """
    
    def __init__(self, basemodel, max_iter = 200, prob_threshold = 0.8):
        self.model = basemodel              # the base model, a classifier from sklearn
        self.max_iter = max_iter            # the number of iteration
        self.prob_threshold = prob_threshold    # the probability threshold
        
    def fit(self, X, y): # -1 for unlabeled
        """Fit base model to the data in a semi-supervised fashion 
        using self training 

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value (-1) for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix will be created from this

        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)

        Returns
        -------
        self : returns an instance of self.
        """
        unlabeledX = X[y==-1, :]        # X_unlabeled
        labeledX = X[y!=-1, :]          # X_labeled
        labeledy = y[y!=-1]             # y_labeled
        
        self.model.fit(labeledX, labeledy)      # train the base model on the labeled data_X_y
        unlabeledy = self.predict(unlabeledX)   # predict the label of the unlabeled data on the trained model
        unlabeledprob = self.predict_proba(unlabeledX)  # get the probability of the unlabeled_X
        unlabeledy_old = []             # a list to store the pred_label
        # re-train, labeling unlabeled instances with model predictions, until convergence
        i = 0
        """
            Conditions:
                1. unlabeled_old
                    1. 首先判断 unlabeled_old 的长度是否为0，初始化的时候
                    2. unlabeledy 与 unlabeledy_old 中有不相同的元素
                2. the iter number
        """
        while (len(unlabeledy_old) == 0 or numpy.any(unlabeledy!=unlabeledy_old)) and i < self.max_iter:
            unlabeledy_old = numpy.copy(unlabeledy)         # copy the unlabeledy to unlabeledy_old
            # 此处的 0 和 1 跟分类的类别数有关系，此处为二分类
            # get the index of the high confidence samples
            uidx = numpy.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold))[0]
            
            # 将置信度较高的样本添加到训练集中，并且用来训练模型
            # vstack add in row
            # hstack add in column
            self.model.fit(numpy.vstack((labeledX, unlabeledX[uidx, :])), numpy.hstack((labeledy, unlabeledy_old[uidx])))
            # update the label of the unlabeledy with the new model
            unlabeledy = self.predict(unlabeledX)
            # get the probability of each unlabeled sample
            unlabeledprob = self.predict_proba(unlabeledX)
            i += 1
        
        if not getattr(self.model, "predict_proba", None):
            # Platt scaling if the model cannot generate predictions itself
            self.plattlr = LR()
            preds = self.model.predict(labeledX)
            self.plattlr.fit( preds.reshape( -1, 1 ), labeledy )
            
        return self
        
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        
        if getattr(self.model, "predict_proba", None):      # Dose the attration exist?
            return self.model.predict_proba(X)              # exist
        else:
            preds = self.model.predict(X)                   # predict the unlabeled_X
            return self.plattlr.predict_proba(preds.reshape( -1, 1 ))   
        
    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        
        return self.model.predict(X)            # function from the sklearn
    
    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    