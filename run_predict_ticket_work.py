#!/usr/bin/env python
# coding: utf-8

# # My Udacity Data Scientist Project 1: predict ticket work

# ## 1. gather data
# In the following, the raw JIRA ticket data is downloaded from the JFrog jira server.
# The data is written into a comma-separated file, which can easily be loaded into a pandas dataframe

# In[19]:


import os
from jira import JIRA
from read_jira_data           import download_data


str_output_file = "data_jfrog.csv"

if not os.path.exists(str_output_file):
    
    str_your_username = "" # insert your login name here
    str_your_password = "" # insert your password here
    
    jira = JIRA( basic_auth=(str_your_username, str_your_password), 
                 options={'server': 'https://www.jfrog.com/jira'})
    
    list_of_jira_project_prefixes = ['RTFACT']
    
    str_status_to_look_for = 'In Progress'
    
    download_data(jira, 
                  str_output_file, 
                  list_of_jira_project_prefixes, 
                  str_status_to_look_for)


# Load the data into a Pandas dataframe and strip (leading/trailing) whitespaces from column names and remove all (leading/trailing) whitespaces from the 'string' entries in the dataframe

# In[20]:


import pandas as pd

df = pd.read_csv('./' + str_output_file)

df.columns = [s.strip() for s in df.columns]

strip_string_lambda = lambda elem: elem.strip()
for columnName in list(df.select_dtypes(include=['object']).columns):
    df[columnName] = df[columnName].apply(strip_string_lambda)


# ## 2. assess

# In[21]:


df.describe()


# ## 3. clean
# Apply "make_nlp_string" to all descriptions and all summaries.
# I will explain this in greater detail, since it's an important step for preparing the string data.

# In[22]:


from nlp_data_preprocessing   import make_nlp_string

lambda_nlp_string = lambda string: make_nlp_string( string )
df["description"] = df["description"].apply(lambda_nlp_string)
df["summary"]     = df["summary"].apply(lambda_nlp_string)


# Detailed explanation of "make_nlp_string"

# In[23]:


import re
import nltk
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


# ## 4. analyze

# In[24]:


import numpy as np

from text_feature_extraction  import extract_tfidf_features
from nlp_data_preprocessing   import categorical_to_quantitative

nMaxFeatures_Description     = 1000    
nMaxFeatures_Summary         = 1000    

df_tfidf_description         = extract_tfidf_features(df["description"], nMaxFeatures_Description)
df_tfidf_description.columns = ["description_" + s for s in df_tfidf_description.columns]

df_tfidf_summary             = extract_tfidf_features(df["summary"], nMaxFeatures_Summary)
df_tfidf_summary.columns     = ["summary_" + s for s in df_tfidf_summary.columns]

df_tickettype                = categorical_to_quantitative( df["issuetype"], "issuetype")

df_tickettype.index          = np.arange(0, len(df_tickettype))

finalDataFrame = pd.concat([df["timeInProgress"], 
                            df_tfidf_description, 
                            df_tfidf_summary, 
                            df_tickettype], 
                            axis=1)

# note, that the invalid data is dropped now so that the words are included in the TFIDF-features
idx_keep             = (finalDataFrame["timeInProgress"] > 0.0) &                        (finalDataFrame["timeInProgress"] < finalDataFrame["timeInProgress"].quantile(q=0.99))
finalDataFrame       = finalDataFrame[idx_keep]
finalDataFrame.index = np.arange(0, len(finalDataFrame))

finalDataFrame["timeInProgress"].hist()

mean_y = finalDataFrame['timeInProgress'].mean()
std_y  = finalDataFrame['timeInProgress'].std()

finalDataFrame_normalized = (finalDataFrame-finalDataFrame.mean()) / finalDataFrame.std()


# In[25]:


df.index


# In[ ]:





# ## 5. model

# In[26]:


import xgboost as xgb
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import mean_squared_error


X_train, X_test, y_train, y_test = train_test_split( finalDataFrame_normalized.drop(columns = ['timeInProgress']), finalDataFrame_normalized['timeInProgress'], test_size=0.3)


np.random.seed(42)

alpha                    = 3
colsample_bytree         = 0.8
colsample_bylevel        = 0.6
colsample_bynode         = 0.9
learning_rate            = 0.25
max_depth                = 5
n_estimators             = 500

    
xg_reg = xgb.XGBRegressor(objective         ='reg:squarederror', 
                          colsample_bytree  = colsample_bytree, 
                          colsample_bylevel = colsample_bylevel,
                          colsample_bynode  = colsample_bynode,
                          learning_rate     = learning_rate,
                          max_depth         = max_depth, 
                          alpha             = alpha, 
                          n_estimators      = n_estimators,
                          seed              = 42,
                          random_state      = 42)

xg_reg.fit(X_train, y_train)

train_mse_rescaled = mean_squared_error(y_train, xg_reg.predict(X_train)) * std_y * std_y
test_mse_rescaled  = mean_squared_error(y_test,  xg_reg.predict(X_test)) * std_y * std_y

print("RMSE on the train set: %5.2f"%(np.sqrt(train_mse_rescaled)))
print("RMSE on the test  set: %5.2f"%(np.sqrt(test_mse_rescaled)))


# ## 6. visualize

# In[27]:


import matplotlib.pyplot as plt
fig = plt.figure(2)
plt.subplot(1,2,1)
plt.xlabel('estimation error [h] (TRAIN set)')
plt.ylabel('count')
df_result = pd.Series((xg_reg.predict(X_train) * std_y) - (y_train * std_y).values)
print( "mean absolute error on the TRAIN set: %5.2f +/- %5.2f"%(df_result.mean(), df_result.std()))
df_result.hist(bins = 100)
plt.subplot(1,2,2)
plt.xlabel('estimation error [h] (TEST set)')
plt.ylabel('count')
df_result = pd.Series((xg_reg.predict(X_test) * std_y) - (y_test * std_y).values)
print( "mean absolute error on the TEST  set: %5.2f +/- %5.2f"%(df_result.mean(), df_result.std()))
df_result.hist(bins = 100)
plt.draw()


# In[ ]:





# In[ ]:


from xgb_analysis import examine_feature_impact

featureLabels, averageImpact, totalImpact = examine_feature_impact( xg_reg, X_train, y_train )


# In[ ]:


sorted_idx = np.argsort(averageImpact)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, averageImpact[sorted_idx], align='center')
plt.yticks(pos, np.array(featureLabels)[sorted_idx])
plt.title('Average Feature Impact')
plt.draw()
plt.show()     


# In[ ]:




