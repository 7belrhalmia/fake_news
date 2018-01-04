# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import feature_selection
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import chi2
class Classifier(BaseEstimator):
    
    def __init__(self):
      

        f5 = feature_selection.RFE(estimator=MultinomialNB(), n_features_to_select=100000, step=100,verbose=1)
        pipeline = Pipeline([
          
                ('rfe_feature_selection', f5),
                ('clf', MultinomialNB()),
                 ])


        self.clf =pipeline



    def fit(self, X, y):
        self.clf.fit(X.toarray(), y)

    def predict(self, X):
        return self.clf.predict(X.toarray())

    def predict_proba(self, X):
        return self.clf.predict_proba(X.toarray())
