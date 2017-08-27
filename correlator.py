# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 21:44:06 2017

@author: Mo
"""

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
        self.outfilename = ""
        self.featureNames = []
        self.dataset = False
        self.relationChain = []
        self.relationChainInfo = []

    
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
           
    
    def init(self, m, n):
        
        # TODO enable range control
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

            # print(command.info())
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
                rands = np.random.normal(command.p1, command.p2,len(ind))
            elif noise == "uniform":
                rands = np.random.uniform(command.p1, command.p2,len(ind))
            elif noise == "none":
                rands = 0
            
            # TODO currently dep value gets overwritten!! --> No multicollinearities makeable
            self.dataset[dep] = command.raiser + command.slope * f + rands
            
        else:
            print("ERROR: At least one correlation partner is not in featurepool.")


    def add(self, dep, ind, com):
        if dep and ind in self.dataset.columns.values:
            newChain = ChainElement(dep, ind, com)
            self.relationChain.append(newChain)
        else:
            print("ERROR: at least one featurename is not in dataset.")


    def clear(self):
        self.relationChainInfo = []
        self.relationChain.clear()
        self.outfilename = ""


    def printRelations(self):
        for rel in self.relationChain:
            print("\t", rel.getInfo())


    # def processNext(self):
    #     rel = self.relationChain.pop()
    #     print("processing:", rel.info())
    #     self.correlate(rel.dependent, rel.independent, rel.command)


    def processRelations(self):

        # correlate all chain elements
        for c in self.relationChain:
            # save filename before processing
            self.relationChainInfo.append(c.getInfo())
            self.outfilename = "_".join(self.relationChainInfo)
            print("correlating: ", c.getInfo())
            self.correlate(c.dependent, c.independent, c.command)
        

    def data(self):
        return self.dataset


    def corTable(self):
        print("-"*30, "\nCorrelation Table")
        print(self.data().corr())


    def saveFile(self):

        path = "generated_data/"
        # outfilepath = "D:\dev\dtfw\Applications\MDSCorrelationDemo\Data/"
        
        m = len(self.data())
        n = len(self.data().columns)

        name = "[{m}x{n}]_".format(m=m,n=n)
        name += "_".join(self.relationChainInfo)
        
        fullname = path + name + ".csv"
        
        self.data().to_csv(fullname)

        print("-"*30)
        print("saved to:", fullname)
        outFileSize = os.path.getsize(fullname)
        print("filesize:", round(outFileSize / 1000000, 2), "MB.")



#################################################
class Command:
    
    def __init__(self):
        # default command

        # possible functional dependencies
        self.ftypes = set(["lin", "exp", "sin", "cos", "log10", "log2", "pow"])
        # TODO make dist list
        self.noises = set(["norm","uni"])

        # basic function type
        self.ftype = "lin"
        self.power = 1

        # adjust function
        self.raiser = 0
        self.slope = 0
        self.shift = 0     # offset for polys

        # noise
        self.noise = "none"
        self.p1 = 0     # mean or uniMin
        self.p2 = 1     # stddev or uniMax
        
        # TODO make command id
        self.info = "default"
        self.updateInfo()
        

    def make(self, slope=0, ftype="lin", noise="none", p1=0, p2=1, raiser=0, shift=0, power=1):
        
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
        if noise == "normal" or "uniform":
            self.noise = noise
            self.p1 = p1
            self.p2 = p2

        self.updateInfo()

 
    def updateInfo(self):
        # slope=0, ftype="lin", noise="none", p1=0, p2=1, raiser=0, shift=0, power=1):
        # (E<--y = 0 + 0.9 * lin(x + 0)^1 + normal(0,1) ---B)
        # self.info = 'y = {0} + {1} * {2}(x + {3})^{4} + {5}({6},{7}) '.format(
        #         self.raiser, self.slope, self.ftype, self.shift, self.power, self.noise, self.mu, self.sig)
        self.info = '([{raiser}]+[{slope},{ftype},{shift},{power}]+[{noise},{p1},{p2}])'.format(
            raiser = self.raiser,
            slope = self.slope,
            ftype = self.ftype,
            shift = self.shift,
            power = self.power,
            noise = self.noise,
            p1 = self.p1,
            p2 = self.p2
            )
                

    def info(self):
        return self.info


#################################################
class ChainElement:
    
    def __init__(self, dep, ind, com):
        self.dependent = dep
        self.independent = ind
        self.command = com

    def getInfo(self):
        ans = "(" + self.dependent + "~" + self.command.info + "~" + self.independent + ")"
        return ans


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


def runTest():

    m = 30    # number of samples
    n = 6     # example: 7 --> [A,...,G]
    
    cor = Correlator()
    cor.init(m, n)

    # "lin", "exp", "sin", "cos", "log10", "log2", "pow"
    # (E<--y = 0 + 0.9 * lin(x + 0)^1 + normal(0,1) ---B)
    com1 = Command()
    com1.make(0.9,"lin","normal")
    com2 = Command()
    com2.make(0.8, "lin","normal")
    com3 = Command()
    com3.make(0.8, "lin", "normal", 0, 1.8)

    # order matters (last in, first out)
    # TODO make graph of relations
    cor.add("A","C", com1)
    cor.add("B","A", com2)
    cor.add("E","B", com3)

    # print(cor.data().head())
    # cor.saveFile()
    # cor.corTable()

    cor.processRelations()

    # cor.corTable()
    cor.saveFile()


    # print(data.columns.values)
    # plt.plot(data)
    # plt.legend()
    # plt.show()

    # vis(data.A, data.B, com1.info)
    #    plt.legend()
    #    plt.show()

runTest()








