# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:33:49 2020

@author: herborts
"""

import pandas as pd
from   sklearn.feature_extraction.text import TfidfVectorizer 


def extract_tfidf_features( pandas_dataframe_column, n_max_features ):
    """extract TFIDF features
    
    TFIDF = term frequency – inverse document frequency

    Parameters
    ----------
    pandas_dataframe_column : pandas dataframe column
        coumn of a dataframe
        
    n_max_features : integer
        max number of features to be considered
        
    Returns
    -------
    pandas dataframe with TFIDF features
    column names represent the feature names
    
    """      
    
    # settings that you use for count vectorizer will go here
    tfidf_vectorizer = TfidfVectorizer( use_idf = True, 
                                        max_features = n_max_features,
                                        ngram_range  = (1,2),
                                        max_df = 0.9,
                                        min_df = 10)
     
    # just send in all your docs here
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform( 
                                                     pandas_dataframe_column )
    
    return pd.DataFrame( tfidf_vectorizer_vectors.todense(), 
                        columns = tfidf_vectorizer.get_feature_names() )
