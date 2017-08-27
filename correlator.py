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
    
    def __init__(self, m=None, n=None):
        self.outfilename = ""
        self.featureNames = []
        self.dataset = False
        self.relationChain = []
        self.relationChainInfo = []

        if m and n != None:
            self.init(m, n)
    
    def makeFeatureNames(self, n):       
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
                f = np.sin(ind + command.shift)
            elif command.ftype == "exp":
                f = np.exp(ind + command.shift)
            elif command.ftype == "cos":
                f = np.cos(ind + command.shift)
            elif command.ftype == "log10":
                f = np.log10(ind + command.shift)
            elif command.ftype == "log2":
                f = np.log2(ind + command.shift)
            elif command.ftype == "pow":
                f = (ind + shift)**command.power

            # set rands
            rands = 0
            
            if command.noise == "normal":
                rands = np.random.normal(command.p1, command.p2,len(ind))
            elif command.noise == "uniform":
                rands = np.random.uniform(command.p1, command.p2,len(ind))
            elif command.noise == "none":
                rands = 0
            
            # TODO currently dep value gets overwritten!! --> No multicollinearities makeable
            self.dataset[dep] = command.raiser + command.slope * f + rands
            
        else:
            print("ERROR: At least one correlation partner is not in featurepool.")


    # def add(self, dep, ind, com):
    #     if dep and ind in self.dataset.columns.values:
    #         newChain = ChainElement(dep, ind, com)
    #         self.relationChain.append(newChain)
    #     else:
    #         print("ERROR: at least one featurename is not in dataset.")


    def add(self, *args):

        # (dep, ind, com)
        # counter = 0
        for arg in args:

            dep = arg[0]
            ind = arg[1]
            com = arg[2]

            if dep and ind in self.dataset.columns.values:
                newChain = ChainElement(dep, ind, com)
                self.relationChain.append(newChain)
            else:
                print("ERROR: {0} or {1} is not in featurenames of data.".format(dep, ind))


    def clear(self):
        self.relationChainInfo = []
        self.relationChain.clear()
        self.outfilename = ""


    def printRelations(self):
        print("Chain to process:")
        count = 1
        for rel in self.relationChain:
            print("\t{0}:".format(count), rel.getInfo())
            count += 1

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
        
        # TODO maybe make separate XML
        fullname = path + name + ".csv"
        
        self.data().to_csv(fullname)

        print("-"*30)
        print("saved to:", fullname)
        outFileSize = os.path.getsize(fullname)
        print("filesize:", round(outFileSize / 1000000, 2), "MB.")



#################################################
class Command:
    
    def __init__(self, slope=0, ftype="lin", noise="none", p1=0, p2=1, raiser=0, shift=0, power=1):

        # possible functional dependencies
        self.ftypes = set(["lin", "exp", "sin", "cos", "log10", "log2", "pow"])
        # TODO make dist list
        self.noises = set(["norm","uni"])
        self.info = "default"

        self.make(slope, ftype, noise, p1, p2, raiser, shift, power)        
        
        # # TODO make command id
        
        

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

    m = 100    # number of samples
    n = 4     # example: 7 --> [A,...,G]
    cor = Correlator(m, n)
    # "lin", "exp", "sin", "cos", "log10", "log2", "pow"
    # first gets worked on first
    chain = [
        ("C","D", Command(0.9,"lin","normal")),
        ("A","C", Command(0.6,"lin","normal",0,.4,power=2)),
    ]

    cor.add(*chain)
    cor.printRelations()

    cor.processRelations()
    cor.saveFile()
    cor.corTable()
    data = cor.data()

    # vis(data["C"], data["A"], cor.relationChain[0].getInfo())
    plt.plot(data, "x")
    plt.show()

runTest()








