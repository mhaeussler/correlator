# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:44:06 2017

@author: Mo
"""

# Correlator


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os


#################################################
# Methods
#################################################
def makeFeatureNames(m):
    """returns feature list with unique dummy names according to m."""
    
    alphabet = tuple(string.ascii_uppercase)
    
    ans = []
    counter = 0;
    counter1 = 0;
    for i in range(m):
        staged = alphabet[i % 26]   # A...Z
        
        if( counter % 26 == 0):
            counter1 += 1
            
        if(staged in ans):
            ans.append(staged*counter1)
        else:
            ans.append(staged)
        
        counter += 1
       
    return ans



def makeOutfilename(filename):
    """Makes hardcoded relative filepath out of given requested name"""
    
    # D:\dev\dtfw\Applications\MDSCorrelationDemo\Data
    # outfilepath = "datasets/"
    outfilepath = "D:\dev\dtfw\Applications\MDSCorrelationDemo\Data/"
    outfileending = ".csv"
    outfilename = str(filename)
    return (outfilepath + outfilename + outfileending)
    


def makeData(numFeatures, numElements, corrFeat1, corrFeat2, strength):
    """Generates synthetic correlated dataset.
    
    Returns:
        pandas data frame.
        bool success value.
    """
    featurenamepool = makeFeatureNames(numFeatures)

    dataset = {}
    
    if corrFeat1 in featurenamepool and corrFeat2 in featurenamepool:

        for index in range(numElements):
            
            innerdic = {}
            
            # initial random sweep
            for name in featurenamepool:
                innerdic[name] = np.random.random()
                
            # correlation sweep
            for name in featurenamepool:
                if name == corrFeat1:
                    innerdic[corrFeat1] = strength * innerdic[corrFeat2]
    
            dataset[index] = innerdic
                
        # make dataframe from dict
        return pd.DataFrame.from_dict(dataset, orient="index"), True
            
    else:
        print("ERROR: At least one correlation partner is not in featurepool.")
        return None, False
    


######################################
# Run
######################################

numFeatures = 4
numElements = 60
corrFeat1 = "A"     # dependent target
corrFeat2 = "B"
corrStrength = 0.8

df, success = makeData(numFeatures, numElements, corrFeat1, corrFeat2, corrStrength)



if success:
    # print(df.head())
    plt.plot(df["A"], "bx")
    
    # write data to csv
    outfilename = makeOutfilename("linCorr_{0}~{1}~{2}_[{3}x{4}]".format(corrFeat1, corrStrength, corrFeat2, numFeatures, numElements))
    df.to_csv(outfilename)
    
    print("file generated:", outfilename)
    
    outFileSize = os.path.getsize(outfilename)
    print("filesize:", round(outFileSize / 1000000, 2), "MB.")
    


















