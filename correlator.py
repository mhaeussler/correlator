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
from collections import deque

ri = np.random.randint
rf = np.random.random


#################################################
# Classes
#################################################
class Correlator:
    
    def __init__(self):
        self.outfilename = "default"
        self.featureNames = []
        self.dataset = False
        self.commandChain = deque()

    
    def makeFeatureNames(self, n):
        """returns feature list with unique dummy names according to m."""
        
        alphabet = tuple(string.ascii_uppercase)
        
        counter = 0;
        counter1 = 0;
        for i in range(n):
            staged = alphabet[i % 26]   # A...Z
            
            if( counter % 26 == 0):
                counter1 += 1
                
            if(staged in self.featureNames):
                self.featureNames.append(staged*counter1)
            else:
                self.featureNames.append(staged)
            
            counter += 1
           
    
    
    def makeRawData(self, m, n):
        """Generates initial data.
            
            m - sample size
            n - number of features

        """
        self.makeFeatureNames(n)
        dataset = {}
        
        for name in self.featureNames:
            # -10.0 ... +10.0
            dataset[name] = np.asarray([10 * rf() for i in range(m)])
            
        self.dataset = pd.DataFrame.from_dict(dataset)



    def correlate(self, dep, ind, command):

        # features to correlate must exist in raw dataset
        if dep and ind in self.dataset.columns.values:

            # print("dependent:\n", self.dataset[dep].head())
            # print("independent:\n", self.dataset[ind].head())
            ind = self.dataset[ind]

            print(command.getInfo())
            # # set function
            f = 0
            if command.ftype == "lin" :
                f = ind + command.shift
            elif command.ftype == "sin":
                f = np.sin(ind + shift)
            elif command.ftype == "exp":
                f = np.exp(ind + shift)
            elif command.ftype == "cos":
                f = np.cos(ind + shift)
            elif command.ftype == "log10":
                f = np.log10(ind + shift)
            elif command.ftype == "log2":
                f = np.log2(ind + shift)
            elif command.ftype == "pow":
                f = (ind + shift)**command.power

            # set rands
            rands = 0
            
            if command.noise == "normal":
                rands = np.random.normal(command.mu, command.sig,len(ind))
            elif noise == "uniform":
                rands = np.random.uniform(command.uniMin, command.uniMax,len(ind))
            elif noise == "none":
                rands = 0
            
            self.dataset[dep] = command.raiser + command.slope * f + rands  
            
        else:
            print("ERROR: At least one correlation partner is not in featurepool.")



    def addCommand(self, command):
        self.commandChain.append(command)

    def addCommmands(self, commands):
        # TODO 
        pass

    def clearCommands(self):
        self.commandChain.clear()

    def showCommands(self):
        for command in self.commandChain:
            print("\t", command.info)

    # correlate two features in raw dataset with given command
    def processOneCorrelation(self, dep, ind, com):
        # TODO
        pass

    def processCommandChain():
        # TODO
        pass


    # Getter and Setter
    def makeOutfilename(filename):
        """Makes hardcoded relative filepath out of given requested name"""
        
        # outfilepath = "datasets/"
        outfilepath = "D:\dev\dtfw\Applications\MDSCorrelationDemo\Data/"
        outfileending = ".csv"
        outfilename = str(filename)
        return (outfilepath + outfilename + outfileending)


    def getDataset(self):
        return self.dataset



#################################################
class Command:
    
    def __init__(self):
        # default command

        # possible functional dependencies
        self.ftypes = set(["lin", "exp", "sin", "cos", "log10", "log2", "pow"])

        # basic function type
        self.ftype = "lin"
        self.power = 1

        # adjust function
        self.raiser = 0
        self.slope = 0
        self.shift = 0     # offset for polys

        # add noise
        self.noise = "none"
        self.mu = 0        # for normal dist
        self.sig = 1     # for normal dist
        self.uniMin = -1   # for uniform dist
        self.uniMax = 1    # for uniform dist
        
        # TODO make command id
        self.setInfo()
        

    
    def setCommand(self, raiser, slope, noise, p1, p2, shift=0, ftype="lin", power=1):
        
        # y = raiser + slope * f(x + shift)^power + noise(p1, p2)

        ftype = ftype.lower()

        if ftype in self.ftypes:
            self.ftype = ftype
        else:
            print("ERROR: funtion type not known.")
            return None

        self.power = power
        self.raiser = raiser
        self.slope = slope
        self.shift = shift

        # set noise
        if noise == "normal":
            self.noise = noise
            self.mu = p1
            self.sig = p2
        if noise == "uniform":
            self.noise = noise
            self.uniMin = p1
            self.uniMax = p2

        self.setInfo()


        
    # Getter Setter
    def setInfo(self):
        self.info = 'y = {0} + {1} * {2}(x + {3})^{4} + {5}({6},{7}) '.format(
                self.raiser, self.slope, self.ftype, self.shift, self.power, self.noise, self.mu, self.sig)
        

    def getInfo(self):
        return self.info



#################################################
class CorrChain:
    pass




#################################################
# Methods
#################################################

def vis(x, y, info):
    xlimits = [-2,9]
    ylimits = [-3,14]
    plt.grid()
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.plot(x,y,"bx", label=info)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    # plt.savefig('lin_normalposcorr.pdf', format="pdf")



######################################
# Run
######################################

# assemble correlation commands
# "lin", "exp", "sin", "cos", "log10", "log2", "pow"
com1 = Command()
com1.setCommand(0,0.7,"normal",0,1)

com2 = Command()
com2.setCommand(0, 0.5, "normal", 0.3, 2)


#####################################
# assemble initial raw dataset
m = 20    # number of rows to generate
n = 4     # number of features
# determine which feature pair gets correlated
# must be in bounds of numFeatures
# example:  numFeatures = 5 --> [A,...,E]
# example:  numFeatures = 7 --> [A,...,G]
correlator = Correlator()
correlator.makeRawData(m, n)

correlator.addCommand(com1)
correlator.addCommand(com2)



# data = correlator.getDataset()
# print(data.head())
# correlator.correlate("A", "C", com1)
# print(data.head())

# print(data.columns.values)
# plt.plot(data)
# plt.legend()
# plt.show()

# print("\nCorrelation Table:")
# print(data.corr())

# print("A-B", pear(data["A"], data["B"]))
# print(pear(data["A"], data["C"]))
# print(pear(data["A"], data["D"]))
# print(pear(data["B"], data["C"]))
# print(pear(data["B"], data["D"]))
# print(pear(data["C"], data["D"]))


# vis(data.A, data.B, com1.info)
#    plt.legend()
#    plt.show()
    
# outfilename = makeOutfilename("lin_[{0}x{1}]_d{2}~{3}~i{4}".format(
#         numElements, numFeatures, dependent, com["ftype"], independent))
# df.to_csv(outfilename)

# print("file generated:", outfilename)

# outFileSize = os.path.getsize(outfilename)
# print("filesize:", round(outFileSize / 1000000, 2), "MB.")
    


















