# -*- coding: utf-8 -*-
from __future__ import print_function

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

from sklearn.base import BaseEstimator
import numpy
import sklearn.metrics
from sklearn.linear_model import LogisticRegression as LR
import nlopt
import scipy.stats

class CPLELearningModel(BaseEstimator):
    """
    Contrastive Pessimistic Likelihood Estimation framework for semi-supervised 
    learning, based on (Loog, 2015). This implementation contains two 
    significant differences to (Loog, 2015):
    - the discriminative likelihood p(y|X), instead of the generative 
    likelihood p(X), is used for optimization
    - apart from `pessimism' (the assumption that the true labels of the 
    unlabeled instances are as adversarial to the likelihood as possible), the 
    optimization objective also tries to increase the likelihood on the labeled
    examples

    This class takes a base model (any scikit learn estimator),
    trains it on the labeled examples, and then uses global optimization to 
    find (soft) label hypotheses for the unlabeled examples in a pessimistic  
    fashion (such that the model log likelihood on the unlabeled data is as  
    small as possible, but the log likelihood on the labeled data is as high 
    as possible)

    See Loog, Marco. "Contrastive Pessimistic Likelihood Estimation for 
    Semi-Supervised Classification." arXiv preprint arXiv:1503.00269 (2015).
    http://arxiv.org/pdf/1503.00269

    Attributes
    ----------
    basemodel : BaseEstimator instance
        Base classifier to be trained on the partially supervised data

    pessimistic : boolean, optional (default=True)
        Whether the label hypotheses for the unlabeled instances should be
        pessimistic (i.e. minimize log likelihood) or optimistic (i.e. 
        maximize log likelihood).
        Pessimistic label hypotheses ensure safety (i.e. the semi-supervised
        solution will not be worse than a model trained on the purely 
        supervised instances)
        
    predict_from_probabilities : boolean, optional (default=False)
        The prediction is calculated from the probabilities if this is True 
        (1 if more likely than the mean predicted probability or 0 otherwise).
        If it is false, the normal base model predictions are used.
        This only affects the predict function. Warning: only set to true if 
        predict will be called with a substantial number of data points
        
    use_sample_weighting : boolean, optional (default=True)
        soft label - value in (0, 1) possibility, hard label - an int value
        Whether to use sample weights (soft labels) for the unlabeled instances.
        Setting this to False allows the use of base classifiers which do not
        support sample weights (but might slow down the optimization)

    max_iter : int, optional (default=3000)
        Maximum number of iterations
        
    verbose : int, optional (default=1)
        Enable verbose output (1 shows progress, 2 shows the detailed log 
        likelihood at every iteration).

    """
    
    def __init__(self, basemodel, pessimistic=True, predict_from_probabilities = False, use_sample_weighting = True, max_iter=3000, verbose = 1):
        self.model = basemodel          # base model from sklearn
        self.pessimistic = pessimistic  # pessimistic ensure the safety
        self.predict_from_probabilities = predict_from_probabilities    # predict from probabilities
        self.use_sample_weighting = use_sample_weighting    # True, get soft label from the base model
        self.max_iter = max_iter        # the max iteration value
        self.verbose = verbose          # enable the verbose output
        
        self.it = 0 # the iteration counter
        self.noimprovementsince = 0 # log likelihood hasn't improved since this number of iterations
        # threshold for iterations without improvements (convergence is assumed when this is reached)
        # self.maxnoimprovementsince = 3 
        self.maxnoimprovementsince = 5
        
        self.buffersize = 200       # size of buffer to check for the convergence
        # buffer for the last few discriminative likelihoods (used to check for convergence)
        self.lastdls = [0]*self.buffersize
        
        # best discriminative likelihood and corresponding soft labels; updated during training
        self.bestdl = numpy.infty       # a value in float type, init with the maximum of float in python
        self.bestlbls = []              # store the best dicriminative likelihood and corresponding soft labels
        
        # unique id
        # generate the unique id, convert the int value (97, 125) to char
        self.id = str(chr(numpy.random.randint(26)+97))+str(chr(numpy.random.randint(26)+97))

    def discriminative_likelihood(self, model, labeledData, labeledy = None, unlabeledData = None, \
                                        unlabeledWeights = None, unlabeledlambda = 1, gradient=[], alpha = 0.01):
        """
            discriminative_likelihood(model, labeledData, labeledy, unlabeledData, unlabeledWeights, unlabeledlambda, gradient, alpha)

            Parameters:
                model: the base model
                labeledData: the train_data of the label samples
                labeledy: the labels of the labeled data
                unlabeledData: the data of the unlabeled samples
                unlabeledWeights: soft label, the unlabeled weight
                
                gradient: the gradient list
        """
        # set the first column value 1 where the original value less than 0.5
        unlabeledy = (unlabeledWeights[:, 0]<0.5)*1         # work as the label of the unlabeled instances
        # the smaller, the more confident
        uweights = numpy.copy(unlabeledWeights[:, 0]) # large prob. for k=0 instances, small prob. for k=1 instances 
        # reflect the confidence
        uweights[unlabeledy==1] = 1-uweights[unlabeledy==1] # subtract from 1 for k=1 instances to reflect confidence
        # the weights of all the instances(labeled and unlabeled)
        weights = numpy.hstack((numpy.ones(len(labeledy)), uweights))
        # the labels of all the instances(labeled and unlabeled)
        labels = numpy.hstack((labeledy, unlabeledy))
        # print("shape of labels is {}".format(labels.shape))
        # print("shape of unlabeledy is {}".format(unlabeledy.shape))
        # print("shape of unlabeledWeights is {}".format(unlabeledWeights.shape))
        
        # fit model on supervised data
        if self.use_sample_weighting:       # True: use the sample_weight during fitting model


            model.fit(numpy.vstack((labeledData, unlabeledData)), labels, sample_weight=weights)



        else:
            model.fit(numpy.vstack((labeledData, unlabeledData)), labels)
        
        # probability of labeled data
        P = model.predict_proba(labeledData)        # labeled data
        
        try:
            # labeled
            # labeled discriminative log likelihood
            # calculate the log_loss, the cross-entropy
            labeledDL = -sklearn.metrics.log_loss(labeledy, P) # log loss of the labeled instances
        except Exception as e:
            print (e)
            P = model.predict_proba(labeledData)        # return the model.predict_proba

        # unlabeled instances probability
        # probability of unlabeled data
        unlabeledP = model.predict_proba(unlabeledData)  # probability
           
        try:
            # unlabeled discriminative log likelihood
            eps = 1e-15         # set the minimum 
            # unlabeled probability
            unlabeledP = numpy.clip(unlabeledP, eps, 1 - eps)       # Clip (limit) the values in an array.

            unlabeledDL = numpy.average((unlabeledWeights*numpy.vstack((1-unlabeledy, unlabeledy)).T*numpy.log(unlabeledP)).sum(axis=1))
        except Exception as e:
            print (e)
            unlabeledP = model.predict_proba(unlabeledData)
        
        if self.pessimistic:            # pessimistic
            # pessimistic: minimize the difference between unlabeled and labeled discriminative likelihood (assume worst case for unknown true labels)
            # minimize
            # unlabeledlambda = 1
            dl = unlabeledlambda * unlabeledDL - labeledDL
        else: 
            # maximize
            # optimistic: minimize negative total discriminative likelihood (i.e. maximize likelihood) 
            dl = - unlabeledlambda * unlabeledDL - labeledDL
        
        return dl
        
    def discriminative_likelihood_objective(self, model, labeledData, labeledy = None, unlabeledData = None, \
                                            unlabeledWeights = None, unlabeledlambda = 1, gradient=[], alpha = 0.01):
        """the discriminative likelihood objective, the optimization objective

            Parameters:
                model: the base model
                labeledData: the train_data of the label samples
                labeledy: the labels of the labeled data
                unlabeledData: the data of the unlabeled samples
                unlabeledWeights: soft label, the unlabeled weight
                
                gradient: the gradient list
        """
        # iteration counter
        if self.it == 0:
            # the last few discriminative likelihoods, for checking convergence
            self.lastdls = [0]*self.buffersize
        
        # calculate the discriminative likelihoods
        # return the discriminative likelihoods
        dl = self.discriminative_likelihood(model, labeledData, labeledy, unlabeledData, unlabeledWeights, unlabeledlambda, gradient, alpha)
        
        self.it += 1
        # update the last buffersize discriminative likelihoods
        self.lastdls[numpy.mod(self.it, len(self.lastdls))] = dl    # np.mod: return the reminder of division
        
        # calculate the difference between the self.lastdls 
        if numpy.mod(self.it, self.buffersize) == 0: # or True:
            # calculate the improvement
            improvement = numpy.mean((self.lastdls[int(len(self.lastdls)/2):])) - numpy.mean((self.lastdls[:int(len(self.lastdls)/2)]))
            # ttest - test for hypothesis that the likelihoods have not changed (i.e. there has been no improvement, and we are close to convergence) 
            _, prob = scipy.stats.ttest_ind(self.lastdls[int(len(self.lastdls)/2):], self.lastdls[:int(len(self.lastdls)/2)])
            
            # if improvement is not certain accoring to t-test...
            # if no improvement, close to convergence
            noimprovement = prob > 0.1 and numpy.mean(self.lastdls[int(len(self.lastdls)/2):]) < numpy.mean(self.lastdls[:int(len(self.lastdls)/2)])
            if noimprovement:
                self.noimprovementsince += 1
                if self.noimprovementsince >= self.maxnoimprovementsince:       # if reach the maxnoimprovementsince
                    # no improvement since a while - converged; exit
                    self.noimprovementsince = 0
                    raise Exception(" converged.") # we need to raise an exception to get NLopt to stop before exceeding the iteration budget
            else:
                self.noimprovementsince = 0
            
            if self.verbose == 2:
                print (self.id,self.it, dl, numpy.mean(self.lastdls), improvement, round(prob, 3), (prob < 0.1))
            elif self.verbose:
                sys.stdout.write(('.' if self.pessimistic else '.') if not noimprovement else 'n')
                      
        if dl < self.bestdl:
            self.bestdl = dl
            self.bestlbls = numpy.copy(unlabeledWeights[:, 0])      # the best dicriminative likelihood and corresponding soft labels
                        
        return dl           # the value 
    
    def fit(self, X, y): # -1 for unlabeled
        unlabeledX = X[y==-1, :]            # the train_data of unlabeled (7451, 122)
        labeledX = X[y!=-1, :]              # train_data of labeled (400, 122)
        labeledy = y[y!=-1]                 # the true label of labeled (400,)        
        
        # the dimensionality of the problem
        M = unlabeledX.shape[0]             # get the number of the labeled samples 7451
        
        # base model training 
        self.model.fit(labeledX, labeledy)  # train the base model 

        # predict the unlabeled samples
        # if the predict_from_probabilities is True, the value is the probabilities of each sample
        unlabeledy = self.predict(unlabeledX)       # predict from the data feature

        
        # re-train, labeling unlabeled instances pessimistically
        
        # pessimistic soft labels ('weights') q for unlabelled points, q=P(k=0|Xu)
        # lambda, create an anonymous function
        f = lambda softlabels, grad=[]: \
                self.discriminative_likelihood_objective(
                                self.model,             # the base model
                                labeledX,               # the train_data of label samples
                                labeledy=labeledy,      # label of the labeled data
                                unlabeledData=unlabeledX,   # the data of unlabeled samples
                                unlabeledWeights=numpy.vstack((softlabels, 1 - numpy.array(softlabels))).T,    # unlabeled weight, soft label
                                gradient=grad           # a list gradient
                                ) #- supLL

        # the same length with the unlabeledy list, 7451
        lblinit = numpy.random.random(len(unlabeledy))      # init the optimization parameters


        # try:
        #     print("+++++++++++++++++++++++++")
        #     self.it = 0                                         # the iteration counter
        #     # the nlopt optimization toolkits
        #     # GN_DIRECT_L_RAND: the NLopt Algorithms
        #     # M: the number of optimization parameters, the number of the unlabeledX.shape[0]
        #     opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, M)
        #     # the bound constraints
        #     opt.set_lower_bounds(numpy.zeros(M))        # set the lower bounds
        #     opt.set_upper_bounds(numpy.ones(M))         # set the upper bounds
        #     opt.set_min_objective(f)                    # set the objective function
        #     opt.set_maxeval(self.max_iter)              # the max times of the optimization
        #     self.bestsoftlbl = opt.optimize(lblinit)    # perform the optimization from the init parameters
        #     print (" max_iter exceeded.")               # print function
        # except Exception as e:                          # deal with exception
        #     print("________________", e)
        #     self.bestsoftlbl = self.bestlbls
            

        print("+++++++++++++++++++++++++")
        self.it = 0                                         # the iteration counter
        # the nlopt optimization toolkits
        # GN_DIRECT_L_RAND: the NLopt Algorithms
        # M: the number of optimization parameters, the number of the unlabeledX.shape[0]
        opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, M)
        # the bound constraints
        opt.set_lower_bounds(numpy.zeros(M))        # set the lower bounds
        opt.set_upper_bounds(numpy.ones(M))         # set the upper bounds
        opt.set_min_objective(f)                    # set the objective function
        opt.set_maxeval(self.max_iter)              # the max times of the optimization
        self.bestsoftlbl = opt.optimize(lblinit)    # perform the optimization from the init parameters
        print (" max_iter exceeded.")               # print function



        if numpy.any(self.bestsoftlbl != self.bestlbls):        # any value statisfy is True
            self.bestsoftlbl = self.bestlbls            # copy
        ll = f(self.bestsoftlbl)

        # return the label
        unlabeledy = (self.bestsoftlbl<0.5)*1           # set the value < 0.5 1, the other are set 0
        # unlabeledy weights
        # the smaller, the better
        uweights = numpy.copy(self.bestsoftlbl) # large prob. for k=0 instances, small prob. for k=1 instances
        # reflect confidence for the instances whose label is 1
        uweights[unlabeledy==1] = 1-uweights[unlabeledy==1] # subtract from 1 for k=1 instances to reflect confidence
        # weights of all the instances
        weights = numpy.hstack((numpy.ones(len(labeledy)), uweights))
        # labels of all the instances
        labels = numpy.hstack((labeledy, unlabeledy))
        if self.use_sample_weighting:           # True, use the weights as the soft label
            # sample_weight: Higher weights force the classifier to put more emphasis on these points.
            self.model.fit(numpy.vstack((labeledX, unlabeledX)), labels, sample_weight=weights)
        else:
            self.model.fit(numpy.vstack((labeledX, unlabeledX)), labels)
        
        # all verbose output
        if self.verbose > 1:
            print ("number of non-one soft labels: ", numpy.sum(self.bestsoftlbl != 1), ", balance:", numpy.sum(self.bestsoftlbl<0.5), " / ", len(self.bestsoftlbl))
            print ("current likelihood: ", ll)
        
        if not getattr(self.model, "predict_proba", None):      # the model doesn't have the predict_proba attribute
            # Platt scaling
            self.plattlr = LR()             # set the LR as the predict_proba
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
        
        if getattr(self.model, "predict_proba", None):
            return self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
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
        
        if self.predict_from_probabilities:
            P = self.predict_proba(X)
            return (P[:, 0] < numpy.average(P[:, 0]))
        else:
            return self.model.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
