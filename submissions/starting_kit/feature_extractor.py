# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
import nltk
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import numpy as np
import scipy.sparse as sp
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter







w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
ps = PorterStemmer()
def joiin(l):
    " ".join(l)
def lemmatize_text(text):
    return [ps.stem(w) for w in w_tokenizer.tokenize(text.decode("utf-8", "replace").lower())]

def polarity_scores(text):
     return sid.polarity_scores(text)
def count_worlds_type(text):
    tagged= nltk.pos_tag(word_tokenize(text.decode("utf-8", "replace")))
    return dict(Counter(tag for word,tag in tagged))


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    

    def fit(self, X_df, y):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``statement``
            column.
        """
      
        transformer = FeatureUnion([
                ('statement_tfidf', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: list(x['statement'].apply(lemmatize_text)), 
                                                  validate=False)),
                            ('extract_field1',
                              FunctionTransformer(lambda x: [" ".join(i) for i in x], 
                                                  validate=False)),
                            ('tfidf', 
                              CountVectorizer(ngram_range=(1,3))),
                           
                            
                            ])),
                 
                
               ('statement_tfidf2', 
                  Pipeline([('extract_field',
                               FunctionTransformer(lambda x: list(x['statement'].apply(polarity_scores)), 
                                                   validate=False)),
                            
                             ('tfidf2', 
                               DictVectorizer())
                           
                            
                             ])),
                
                ('search_term_tfidf2', 
                  Pipeline([('extract_field2',
                              FunctionTransformer(lambda x: x['researched_by'].astype('U'), 
                                                  validate=False)),
                           
                            ('tfidf2', 
                              CountVectorizer(ngram_range=(1,3)))
                            
                            ])),
                ('search_term_tfidf3', 
                  Pipeline([('extract_field3',
                              FunctionTransformer(lambda x: x['edited_by'].astype('U'), 
                                                  validate=False)),
                           
                            ('tfidf3', 
                              CountVectorizer(ngram_range=(1,3)))
                            
                            ])),
                ('search_term_tfidf4', 
                  Pipeline([('extract_field4',
                              FunctionTransformer(lambda x: x['job'].astype('U'), 
                                                  validate=False)),
                           
                            ('tfidf4', 
                              CountVectorizer(ngram_range=(1,5)))
                            
                            ])), 
                ('search_term_tfidf5', 
                  Pipeline([('extract_field5',
                              FunctionTransformer(lambda x: x['source'].astype('U'), 
                                                  validate=False)),
                           
                            ('tfidf5', 
                              CountVectorizer(ngram_range=(1,5)))
                            
                            ])),    
                ('search_term_tfidf6', 
                  Pipeline([('extract_field6',
                              FunctionTransformer(lambda x: x['state'].astype('U'), 
                                                  validate=False)),
                           
                            ('tfidf6', 
                              CountVectorizer(ngram_range=(1,3)))
                            
                            ])),
                ('search_term_tfidf7', 
                  Pipeline([('extract_field7',
                              FunctionTransformer(lambda x: x['subjects'].astype('U'), 
                                                  validate=False)),
                           
                            ('tfidf7', 
                              CountVectorizer(ngram_range=(1,5)))
                            
                            ]))
            
                ]) 
        
        self._vecorizer=transformer.fit(X_df)
        
        return self


        
        
        



    def transform(self, X_df):

        bag_of_words = self._vecorizer.transform(X_df)
       
        return bag_of_words


