# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
class Classifier(BaseEstimator):
    
    def __init__(self):
        parameters={
           'learning_rate': ["adaptive"],
           'hidden_layer_sizes': [(256,)],
           'activation': [ "tanh"],
            'solver' : ['sgd']
           }
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        #param_grid = {'alpha': [1, 0.1,0.9,0.8,0.3, 0.01, 0.001, 0.0001, 0.00001] }
        #param_grid2 = {'C': [1000,100,10,1, 0.001, 0.0001, 0.00001] }
        #params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
        self.clf = OneVsRestClassifier(MultinomialNB())
        #clf2 = GridSearchCV(LogisticRegression(),param_grid=param_grid2, n_jobs=-1)


        #clf3 = GridSearchCV(svm.SVC(probability=True),param_grid=tuned_parameters, n_jobs=-1)
        
        #self.clf=VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),('rf2', clf3)], voting="soft")
    def fit(self, X, y):
        self.clf.fit(X.toarray(), y)

    def predict(self, X):
        return self.clf.predict(X.toarray())

    def predict_proba(self, X):
        return self.clf.predict_proba(X.toarray())
