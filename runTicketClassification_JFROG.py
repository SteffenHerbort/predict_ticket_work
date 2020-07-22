# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:01:18 2020

@author: herborts
"""

# gather, assess, clean, analyze, model, visualize


import os
import platform
import random

import numpy as np
np.random.seed(42)

import pandas            as pd
import matplotlib.pyplot as plt
import datetime          as datetime

from jira                    import JIRA
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error, explained_variance_score

# XGB trees
import xgboost as xgb

from read_jira_data           import download_data
from nlp_data_preprocessing   import make_nlp_string, categorical_to_quantitative
from text_feature_extraction  import extract_tfidf_features
from xgb_analysis             import examine_feature_impact
from utilities                import get_parameter_combinations

if ( platform.system() == "Windows" ):
    strFolder  = "U:/herborts/privat/dev/predict_ticket_work"

elif( platform.system() == "Linux" ):
    strFolder  = "/import/mvtec/home/herborts/privat/dev/predict_ticket_work"
    

os.chdir(strFolder)

str_output_file = "data_jfrog.csv"




# download data if it's not there (yet)
if not os.path.exists(str_output_file):
    print("data does not exist -> download from jira server")
    
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
    
    
print("read data...", end="")
df = pd.read_csv('./' + str_output_file)
print("OK")

# strip (leading/trailing) whitespaces from column names
df.columns = [s.strip() for s in df.columns]

#only use the ARTIFACTORY tickets
df = df[df["project_prefix"] == 'RTFACT']    
 
# strip (leading/trailing) whitespaces from all strings in the dataframe
strip_string_lambda = lambda elem: elem.strip()
for columnName in list(df.select_dtypes(include=['object']).columns):
    df[columnName] = df[columnName].apply(strip_string_lambda)


print("text preprocessing for all descriptions + summaries...", end="", flush=True)
lambda_nlp_string = lambda string: make_nlp_string( string )
df["description"] = df["description"].apply(lambda_nlp_string)
df["summary"]     = df["summary"].apply(lambda_nlp_string)
print("OK")
    


#nMaxFeatures_Description        : 1000    
#nMaxFeatures_Summary            : 1000    
#alpha                           : 3       
#colsample_bytree                : 0.8     
#colsample_bylevel               : 0.6     
#colsample_bynode                : 0.9     
#learning_rate                   : 0.25    
#max_depth                       : 5       
#n_estimators                    : 500    



param_range = {}
param_range["nMaxFeatures_Description"] = np.array([10, 20, 50, 100, 200, 300, 400, 500, 1000]) 
param_range["nMaxFeatures_Summary"]     = np.array([10, 20, 50, 100, 200, 300, 400, 500, 1000])  
param_range["alpha"]                    = np.array([0.1, 1, 10, 100]) 
param_range["colsample_bytree"]         = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
param_range["colsample_bylevel"]        = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
param_range["colsample_bynode"]         = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
param_range["learning_rate"]            = np.array([0.15, 0.2, 0.25, 0.3])
param_range["max_depth"]                = np.array([3, 5, 7, 9, 12])
param_range["n_estimators"]             = np.array([10, 50, 100, 200, 300, 400, 500, 750, 1000, 2000, 5000])

#param_range = {}
#param_range["nMaxFeatures_Description"] = np.array([1000]) 
#param_range["nMaxFeatures_Summary"]     = np.array([1000])  
#param_range["alpha"]                    = np.array([3])
#param_range["colsample_bytree"]         = np.array([0.8])
#param_range["colsample_bylevel"]        = np.array([0.6])
#param_range["colsample_bynode"]         = np.array([0.9])
#param_range["learning_rate"]            = np.array([0.25])
#param_range["max_depth"]                = np.array([5])
#param_range["n_estimators"]             = np.array([500])


listParamRange = get_parameter_combinations( param_range )

best = {}
best["best_test+train"]             = 1e9
best["best_sum_explained_variance"] = -1e9

best["test_mse"]                    = 1e9
best["test_mse_rescaled"]           = 1e9
best["test_explained_variance"]     = -1.0

best["train_mse"]                   = 1e9
best["train_mse_rescaled"]          = 1e9
best["train_explained_variance"]    = -1e9

best["params"]             = {}
best["model"]              = [] 

random.shuffle(listParamRange)


for paramSet in listParamRange:
    
    nMaxFeatures_Description = paramSet["nMaxFeatures_Description"]
    nMaxFeatures_Summary     = paramSet["nMaxFeatures_Summary"]
    alpha                    = paramSet["alpha"]
    colsample_bytree         = paramSet["colsample_bytree"]
    colsample_bylevel        = paramSet["colsample_bylevel"]
    colsample_bynode         = paramSet["colsample_bynode"]
    learning_rate            = paramSet["learning_rate"]
    max_depth                = paramSet["max_depth"]
    n_estimators             = paramSet["n_estimators"]
    

    # extract 'description' features
    df_tfidf_description         = extract_tfidf_features(df["description"], nMaxFeatures_Description)
    df_tfidf_description.columns = ["description_" + s for s in df_tfidf_description.columns]

    
    # extract 'summary' features
    df_tfidf_summary             = extract_tfidf_features(df["summary"], nMaxFeatures_Summary)
    df_tfidf_summary.columns     = ["summary_" + s for s in df_tfidf_summary.columns]

    # extract 'tickettype' features
    df_tickettype                = categorical_to_quantitative(
                                                            df["issuetype"], 
                                                            "issuetype")
    df_tickettype.index          = np.arange(0, len(df_tickettype))
    
    
    finalDataFrame = pd.concat([df["timeInProgress"], df_tfidf_description, df_tfidf_summary, df_tickettype], axis=1)
    
    # drop data where "y" is 0 or an outlier
    # outliers are above finalDataFrame["timeInProgress"].quantile(q=0.96)  ~~  200h
    idx_keep = (finalDataFrame["timeInProgress"] > 0.0001) & (finalDataFrame["timeInProgress"] < 200.0)
    finalDataFrame = finalDataFrame[idx_keep]
    
    
    
    if True:
        
        mean_y = finalDataFrame['timeInProgress'].mean()
        std_y  = finalDataFrame['timeInProgress'].std()
        
        finalDataFrame_normalized = (finalDataFrame-finalDataFrame.mean()) / finalDataFrame.std()      
                
    else:
        finalDataFrame_normalized = finalDataFrame
        mean_y = 0.0
        std_y = 1.0
    
    X_train, X_test, y_train, y_test = train_test_split( 
        finalDataFrame_normalized.drop(columns = ['timeInProgress']), 
        finalDataFrame_normalized['timeInProgress'], 
        test_size=0.2, 
        random_state=42)
    
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', 
                              colsample_bytree  = colsample_bytree, 
                              colsample_bylevel = colsample_bylevel,
                              colsample_bynode  = colsample_bynode,
                              learning_rate     = learning_rate,
                              max_depth         = max_depth, 
                              alpha             = alpha, 
                              n_estimators      = n_estimators,
                              random_state      = 42)
    
    print("train %d trees..."%(n_estimators), end="", flush=True)
    xg_reg.fit(X_train, y_train)
    print("OK", flush=True)
    

    
    train_mse                = mean_squared_error(y_train, xg_reg.predict(X_train))
    test_mse                 = mean_squared_error(y_test,  xg_reg.predict(X_test))
    train_mse_rescaled       = mean_squared_error(y_train, xg_reg.predict(X_train)) * std_y * std_y
    test_mse_rescaled        = mean_squared_error(y_test,  xg_reg.predict(X_test)) * std_y * std_y   
    train_explained_variance = explained_variance_score(y_train, xg_reg.predict(X_train))
    test_explained_variance  = explained_variance_score(y_test, xg_reg.predict(X_test))
    
    #if test_mse+train_mse < best["best_test+train"]:
    #if train_explained_variance + test_explained_variance > best["best_sum_explained_variance"]:
    if test_explained_variance > best["test_explained_variance"]:
        
        
        print("examine feature importances...", end="", flush=True)
        featureLabels, averageImpact, totalImpact = examine_feature_impact( xg_reg, X_train, y_train )
        print("OK", flush=True)    
            
       
        
        if False:
            train_predictions = mean_y + xg_reg.predict(X_train) * std_y
            train_truth       = mean_y + y_train                 * std_y
            test_predictions  = mean_y + xg_reg.predict(X_test)  * std_y
            test_truth        = mean_y + y_test                  * std_y
            
            plt.figure(12)
            
            plt.subplot(1,2,1)
            idx = np.argsort( train_truth.values ).astype(int)
            plt.barh(y=np.arange(0, len(train_truth.values)), width=train_truth.values[idx])
            plt.barh(y=np.arange(0, len(train_predictions)),  width=train_predictions[idx]*(-1))
    
            plt.subplot(1,2,2)
            idx = np.argsort( test_truth.values ).astype(int)
            plt.barh(y=np.arange(0, len(test_truth.values)), width=test_truth.values[idx])
            plt.barh(y=np.arange(0, len(test_predictions)),  width=test_predictions[idx]*(-1))
            
            plt.show()        
        
       
        best["best_sum_explained_variance"] = train_explained_variance + test_explained_variance
        best["best_test+train"]             = test_mse + train_mse
        
        best["train_mse"]                   = train_mse
        best["train_mse_rescaled"]          = train_mse_rescaled
        best["train_explained_variance"]    = train_explained_variance
        
        best["test_mse"]                    = test_mse
        best["test_mse_rescaled"]           = test_mse_rescaled
        best["test_explained_variance"]     = test_explained_variance
        
        best["params"]                      = paramSet
        best["model"]                       = xg_reg
        
        
        strPrefix = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S_')
        with open( "./results_jfrog/" + strPrefix + "report.txt", "wb") as f:
            f.write( "scores:\n".encode() )
            f.write( ("    test_mse+train_mse       : %12.3f \n"%(test_mse+train_mse)).encode())
            f.write( ("    sum_explained_variance   : %12.3f \n"%(train_explained_variance + test_explained_variance)).encode())
            f.write( ("    train_mse                : %12.3f \n"%(train_mse)).encode())
            f.write( ("    train_mse_rescaled       : %12.3f \n"%(train_mse_rescaled)).encode())
            f.write( ("    train_explained_variance : %12.3f \n"%(train_explained_variance)).encode())
            f.write( ("    test_mse                 : %12.3f \n"%(test_mse)).encode())
            f.write( ("    test_mse_rescaled        : %12.3f \n"%(test_mse_rescaled)).encode())
            f.write( ("    test_explained_variance  : %12.3f \n"%(test_explained_variance)).encode())
            f.write( "\n".encode() ) 
            
            f.write( "parameters:\n".encode() )
            for key in paramSet.keys():
                f.write( (key.ljust(32) + ": " + str(paramSet[key]).ljust(8) + "\n").encode() )
            f.write( "\n".encode() ) 
            
            f.write( "featureLabel, averageImpact, totalImpact\n".encode() )
            for i, elem in enumerate(featureLabels):
                string = featureLabels[i].ljust(32) + "  :  %18.6f  :  %18.6f \n"%(totalImpact[i], averageImpact[i])
                f.write( string.encode() )    
        
        
        plt.close("all")
        
        fig = plt.figure(1)

        feature_importance = xg_reg.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(finalDataFrame.columns)[sorted_idx])
        plt.title('Feature Importance (MDI)')
        plt.draw()
        plt.show()        
        
        
        fig = plt.figure(2)
        plt.subplot(1,2,1)
        plt.xlabel('error (TRAIN set)')
        plt.ylabel('count')
        df_result = pd.Series((xg_reg.predict(X_train) * std_y) - (y_train * std_y).values)
        df_result.describe()
        df_result.hist(bins = 100)
        plt.subplot(1,2,2)
        plt.xlabel('error (TEST set)')
        plt.ylabel('count')
        df_result = pd.Series((xg_reg.predict(X_test) * std_y) - (y_test * std_y).values)
        df_result.describe()
        df_result.hist(bins = 100)
        plt.draw()

        if False:
            for idx, strType in enumerate(["weight", "gain", "total_gain"]): #enumerate(["weight", "gain", "cover", "total_gain", "total_cover"]):
                fig = plt.figure(3 + idx)
                fig.set_size_inches(5, 12)
                feature_important = best["model"].get_booster().get_score(importance_type=strType)
                keys   = list(feature_important.keys())
                values = list(feature_important.values())
                data = pd.DataFrame(data=values, index=keys, columns=[strType]).sort_values(by = strType, ascending=True).head(50)
                data.plot(kind='barh', ax=fig.gca())
                plt.title(strType)
                plt.subplots_adjust(left=0.4, right=0.95, top=0.95, bottom=0.05)
                plt.draw()
                plt.pause(0.001)

    
    print("                         current params             best params:")
    for key in paramSet.keys():
        print(key.ljust(32) + ": " + str(paramSet[key]).ljust(8) + "        best: " + str(best["params"][key]))
    
    print("    train+test set mse   = %12.3f             best = %12.3f"%(test_mse+train_mse,                                                   best["best_test+train"]))    
    print("         train set mse   = %12.3f             best = %12.3f"%(mean_squared_error(y_train, xg_reg.predict(X_train)),                 best["train_mse"]))
    print("rescaled train set mse   = %12.3f             best = %12.3f"%(mean_squared_error(y_train, xg_reg.predict(X_train)) * std_y * std_y, best["train_mse_rescaled"]))
    print("         test  set mse   = %12.3f             best = %12.3f"%(mean_squared_error(y_test, xg_reg.predict(X_test)),                   best["test_mse"]))
    print("rescaled test  set mse   = %12.3f             best = %12.3f"%(mean_squared_error(y_test, xg_reg.predict(X_test)) * std_y * std_y,   best["test_mse_rescaled"]))
    print("sum_explained_variance   = %12.3f             best = %12.3f"%(train_explained_variance + test_explained_variance,                   best["best_sum_explained_variance"]))
    print("train_explained_variance = %12.3f             best = %12.3f"%(train_explained_variance,                                             best["train_explained_variance"]))
    print("test_explained_variance  = %12.3f             best = %12.3f"%(test_explained_variance,                                              best["test_explained_variance"]))
    print("")
             

        
        
    
    
    
    
    
        
 