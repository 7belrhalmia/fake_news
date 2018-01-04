# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import feature_selection
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel

class Classifier(BaseEstimator):
    
    def __init__(self):
      
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        #param_grid = {'alpha': [1, 0.1,0.9,0.8,0.3, 0.01, 0.001, 0.0001, 0.00001] }
        #param_grid2 = {'C': [1000,100,10,1, 0.001, 0.0001, 0.00001] }
        #params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
        #f5 = feature_selection.RFE(estimator=MultinomialNB(), n_features_to_select=180386, step=2,verbose=1)
        #pipeline = Pipeline([
          
              #  ('rfe_feature_selection', f5),
               # ('clf', MultinomialNB()),
                #  ])
        
        pipeline = Pipeline([
          
                ('rfe_feature_selection', SelectFromModel(LogisticRegression(C=1000, penalty="l2"))),
                ('clf', OneVsRestClassifier(MultinomialNB())),
                  ])

        self.clf =pipeline 


        #clf2 = GridSearchCV(LogisticRegression(),param_grid=param_grid2, n_jobs=-1)


        #clf3 = GridSearchCV(svm.SVC(probability=True),param_grid=tuned_parameters, n_jobs=-1)
        
        #self.clf=VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),('rf2', clf3)], voting="soft")
    def fit(self, X, y):
        self.clf.fit(X.toarray(), y)

    def predict(self, X):
        return self.clf.predict(X.toarray())

    def predict_proba(self, X):
        return self.clf.predict_proba(X.toarray())
