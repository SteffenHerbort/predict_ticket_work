# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:36:08 2020

@author: herborts
"""
import numpy as np

def examine_feature_impact( xg_reg, X_train, y_train, order = 1 ):
    """examines the impact (+/-) of features in a decision tree
    
    The idea is: 1) Remove feature(s) from a sample (set its vector to 0)
                 2) Determine, how the prediction value changes ("impact")
                 3) Do this for all features
                 

    Parameters
    ----------
    xg_reg : XGB regressor
        trained model
        
    X_train : training data matrix
        training data features

    y_train : target value vector
        training data target
    
    order   : integer
        number of features that are removed at the same time
        1 (default)  : one feature after the other is removed an their impact
                       is tested
        2            : two distinct features are removed and their impact 
                       is tested
        3 (or higher): not available!              
        
        
    Returns
    -------
    featureLabelsSorted : list of strings
        labels of each features, sorted by the impact (descending)
    averageImpactSorted
        average impact of each feature, sorted by the impact (descending)
    totalImpactSorted
        total/summed impact of each feature, sorted by the impact (descending)
    
    """         
    
    featureLabels = []
    totalImpact   = []
    
    y_pred     = xg_reg.predict(X_train)
    fReference = (y_pred - y_train).sum()
    
    if order == 1:
    
        for columnName in X_train.columns:
            
            X = X_train.copy()
            X[columnName] = 0.0
            y = xg_reg.predict(X)
            
            fDiffSum = (y - y_train).sum()
            
            # "-1" is needed, since the impact is determined
            # by what happens when the feature is MISSING
            totalImpactOfFeature = (-1) * (fDiffSum - fReference)
            
            featureLabels.append(columnName)
            totalImpact.append( totalImpactOfFeature )
            
    elif order == 2:
        
        for idx1, columnName1 in enumerate(X_train.columns):
            for idx2, columnName2 in enumerate(X_train.columns):
                if idx2 <= idx1:
                    continue
                
                X = X_train.copy()
                X[columnName1] = 0.0
                X[columnName2] = 0.0
                y = xg_reg.predict(X)
                
                fDiffSum = (y - y_train).sum()
                
                # "-1" is needed, since the impact is determined
                # by what happens when the feature is MISSING
                totalImpactOfFeature = (-1) * (fDiffSum - fReference)
                
                featureLabels.append(columnName1 + " & " + columnName2)
                totalImpact.append( totalImpactOfFeature )               
                
            
        
    else:
        raise ValueError(("Parameter 'order' can only be '1' or '2', " \
                         "but yours is " + str(order)))
        
    idx = np.argsort(totalImpact)[::-1].astype(int)
    
    featureLabelsSorted = list( np.array(featureLabels)[idx] )
    totalImpactSorted   = list( np.array(totalImpact)[idx] )
    averageImpactSorted = list( np.array(totalImpact)[idx] / len(totalImpact))

        
    return featureLabelsSorted, averageImpactSorted, totalImpactSorted
