# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:16:28 2020

@author: SteffenH
"""



import re

import pandas as pd


import nltk

# resources only need to be downloaded once
for resource in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger", "snowball_data"]:
    try:
        nltk.data.find('tokenizers/' + resource)
    except LookupError:
        nltk.download(resource, quiet = True)


from nltk.tokenize      import word_tokenize

from nltk.corpus        import stopwords

from nltk.stem.snowball import SnowballStemmer

import langdetect


def make_nlp_string( str_document ):
    """preprocessing of a document

    1) removes line endings
    2) removes consecutive white spaces
    3) removes special characters
    4) removes stop words
    5) performs word stemming / lemmatization

    Parameters
    ----------
    str_document : string
        string of possibly several sentences that form a document
        
    Returns
    -------
    whitespace-delimited string of the preprocessed words
    
    """    
    
    str_language = "english"
    
    try:
        tmp = langdetect.detect( str_document )
        if tmp == "de":
            str_language = "german"
        else:
            str_language = "english"
    except:
        pass #langdetect not successful. Fallback: 'english'
    
        
    # remove line endings
    string_tmp = str_document.replace("\n", " ").replace("\r", " ")
    
    # remove non-alphanumeric characters
    string_tmp = re.sub('[^a-zA-ZäöüÄÖÜß ]', ' ', string_tmp)
    
    # remove consecutive whitespaces
    string_tmp = re.sub(' +', ' ', string_tmp)
    
    # make lower case
    string_tmp = string_tmp.lower()
    
    # break into words
    word_tokens = [w for w in word_tokenize(string_tmp) if len(w) > 1]
    
    # remove stop words
    stop_words = stopwords.words( str_language )
    
    # break into individual words
    listWords = [w for w in word_tokens if not w in stop_words]
    
    # perform word stemming / lemmatization    
    if str_language == 'german':
        stemmer = SnowballStemmer("german")    
    elif str_language == 'english':
        stemmer = SnowballStemmer("english")

    for idx, word in enumerate(listWords):
        listWords[idx] = stemmer.stem( listWords[idx] )
             
    return " ".join(listWords)
    




def categorical_to_quantitative(pandas_dataframe_column, column_name_prefix):
    """convenience function for transforming categorical data into quantitative

    Parameters
    ----------
    pandas_dataframe_column : pandas dataframe column
        simply a column of a dataframe
        
    column_name_prefix : string
        this prefix will be added to all new columns' names
        
    Returns
    -------
    pandas dataframe where each category is one-hot-coded
    
    """        
    return pd.get_dummies(pandas_dataframe_column, 
                          prefix=column_name_prefix, 
                          prefix_sep = '_', 
                          drop_first = False)

