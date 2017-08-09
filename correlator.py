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

ri = np.random.randint
rf = np.random.random


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



def makeCommand(raiser, slope, noise, p1, p2, shift=0, ftype="lin", power=1):
    """Command builder for convenience.
    
    Returns:
        command - dict.
    """
    ftypes = set(["lin", "exp", "sin", "cos", "log10", "log2", "pow"])
        
    if ftype not in ftypes:
        print("ERROR: funtion type not known. Assume linear...")
        ftype = ""
    
    
    com = {

        "raiser": raiser,
        "slope": slope,
        "noise": noise,
        "shift": shift,     # offset for polys
        "mu": p1,        # for normal dist
        "sig": p2,     # for normal dist
        "uniMin": p1,   # for uniform dist
        "uniMax": p2,    # for uniform dist
        
        "ftype": ftype,
        "power": power
        }
    
    # (0, 1, "normal", 0, 1, 0, "lin")
    info = 'y = {0} + {1} * {2}(x + {3})^{4} + {5}({6},{7}) '.format(
            com["raiser"], com["slope"], com["ftype"], com["shift"], com["power"], com["noise"], p1, p2)
    return com, info



def correlate(x, com):
    """Generates y values for given x array and command.
    
    Returns:
        y - array."""
        
    # make function type
    ftype = com["ftype"].lower()
    k = com["shift"]
    f = 0
    
    if ftype == "lin":
        f = x + k
    elif ftype == "sin":
        f = np.sin(x + k)
    elif ftype == "exp":
        f = np.exp(x + k)
    elif ftype == "cos":
        f = np.cos(x + k)
    elif ftype == "log10":
        f = np.log10(x + k)
    elif ftype == "log2":
        f = np.log2(x + k)
    elif ftype == "pow":
        f = (x + k)**com["power"]

    
    # make random values
    rands = 0
    noise = com["noise"].lower()
    
    if noise == "normal":
        rands = np.random.normal(com["mu"],com["sig"],len(x))
    elif noise == "uniform":
        rands = np.random.uniform(com["uniMin"],com["uniMax"],len(x))
    elif noise == "none":
        rands = 0
    else:
        print("ERROR: noise-type unknown.")
    
    # assemble command
    y  = com["raiser"] + com["slope"] * f + rands 
    return y


def makeOutfilename(filename):
    """Makes hardcoded relative filepath out of given requested name"""
    
    # D:\dev\dtfw\Applications\MDSCorrelationDemo\Data
    # outfilepath = "datasets/"
    outfilepath = "D:\dev\dtfw\Applications\MDSCorrelationDemo\Data/"
    outfileending = ".csv"
    outfilename = str(filename)
    return (outfilepath + outfilename + outfileending)
    


def vis(x, y, info):
    xlimits = [-2,9]
    ylimits = [-3,14]
    plt.grid()
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.plot(x,y,"bx", label=info)
    plt.legend()
    plt.show()   
    
    

def makeData(numFeatures, numElements, corrFeat1, corrFeat2, command):
    """Generates synthetic correlated dataset.
    
    Returns:
        pandas data frame.
        bool success value.
    """
    featurenamepool = makeFeatureNames(numFeatures)

    dataset = {}
    
    if corrFeat1 in featurenamepool and corrFeat2 in featurenamepool:

        for name in featurenamepool:
            dataset[name] = np.asarray([10 * rf() for i in range(numElements)])
            
        # correlate
        dataset[corrFeat1] = correlate(dataset[corrFeat2], command)
        
                
        return pd.DataFrame.from_dict(dataset), True
            
    else:
        print("ERROR: At least one correlation partner is not in featurepool.")
        return None, False
    




######################################
# Run
######################################

# assemble the correlation command
raiser = 0
slope = 0.9
noise = "normal"
shift = 0
mu = 0
sig = 0.5
uniMin = -1
uniMax = 1
ftype = "lin"   # "lin", "exp", "sin", "cos", "log10", "log2", "pow"
power = 1

com, info = makeCommand(raiser, slope, noise, mu, sig, shift, ftype, power)


# assemble dataset
numFeatures = 4
numElements = 40
dependent = "A"     # dependent target
independent = "C"

df, success = makeData(numFeatures, numElements, dependent, independent, com)


if success:
    # print(df.head())
    # df.plot("A", "C", kind="scatter")
    
    outfilename = makeOutfilename("linCorr_[{0}x{1}]_d{2}~{3}~i{4}".format(
            numElements, numFeatures, dependent, com["ftype"], independent))
    df.to_csv(outfilename)
    
    print("file generated:", outfilename)
    
    outFileSize = os.path.getsize(outfilename)
    print("filesize:", round(outFileSize / 1000000, 2), "MB.")
    


















