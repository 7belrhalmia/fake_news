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
def lemmatize_text2(text):
    k=[]
    for i in text :
        k.append(i.values())
def polarity_scores(text):
     return sid.polarity_scores(text)
def lemmatize_text234(text):
    text=text.replace('"','').replace(',','').replace(':','').replace('$','').replace(')','').replace('(','').replace('#','').replace('$','')
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
        ###C=[]
        ##self.y=y
        #X= X_df.reset_index().values
        #for n,i in zip(X,y):
        #    C.append(np.append(n,i))
        #X=np.array(C)
        #print X[:3]
        #l=[]
        #for n,i in enumerate(X):
           #x=i[-1].replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace(",", " ")
           #x=x.split(" ")
           #for r in x:
            #  l.append(list(np.append(X[n][:-1],r)))  
        #l=np.array(l)
        
        
            
        #count = CountVectorizer(ngram_range=(1,3))
        ##self._vecorizer1= count.fit(X_df.state.values.astype('U'))
        #count = CountVectorizer(ngram_range=(1,5))
       #self._vecorizer2= count.fit(X_df.edited_by.values.astype('U'))
        #count = CountVectorizer(ngram_range=(1,3))
        #self._vecorizer3= count.fit(X_df.researched_by.values.astype('U'))
        #count = CountVectorizer(ngram_range=(1,5))
        #self._vecorizer4= count.fit(X_df.source.values.astype('U'))
        #count = CountVectorizer(stop_words='english',ngram_range=(1,20))
        #self._vecorizer5= count.fit(X_df.statement.values.astype('U'))
        #count = CountVectorizer(ngram_range=(1,10),lowercase=True)
        
        
        
        #self._vecorizer6= count.fit(X_df.subjects.values.astype('U'))
        #count = CountVectorizer(ngram_range=(1,1))
        #self._vecorizer7= count.fit(X_df.date.values.astype('U'))
        transformer = FeatureUnion([
                ('search_term_tfidf', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: list(x['statement'].apply(lemmatize_text)), 
                                                  validate=False)),
                            ('extract_field1',
                              FunctionTransformer(lambda x: [" ".join(i) for i in x], 
                                                  validate=False)),
                            ('tfidf', 
                              CountVectorizer(ngram_range=(1,1))),
                           
                            
                            ])),
                 
                
             #('search_term_tfidf234', 
                  #Pipeline([('extract_field',
                            #  FunctionTransformer(lambda x: list(x['statement'].apply(polarity_scores)), 
                                              #    validate=False)),
                            
                            #('tfidf2', 
                            #  DictVectorizer())
                           
                            
                            #])),
                
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
        
        self._vecorizer5=transformer.fit(X_df)
        
        return self


        
        
        



    def transform(self, X_df):
        
        

        #bag_of_words1 = self._vecorizer1.transform(X_df.state.values.astype('U'))
        #bag_of_words2 = self._vecorizer2.transform(X_df.edited_by.values.astype('U'))
        #bag_of_words3 = self._vecorizer3.transform(X_df.researched_by.values.astype('U'))
        #bag_of_words4 = self._vecorizer4.transform(X_df.source.values.astype('U'))
        bag_of_words5 = self._vecorizer5.transform(X_df)
        #bag_of_words6 = self._vecorizer6.transform(X_df.subjects.values.astype('U'))
        #bag_of_words7 = self._vecorizer7.transform(X_df.date.values.astype('U'))

        



        return bag_of_words5
        #return sp.hstack([ bag_of_words1,bag_of_words2,bag_of_words3,bag_of_words4,bag_of_words5,bag_of_words6], format='csr')


