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
        self.chainInfo = []
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
            rands = 0
            if command.noise == "n":
                rands = np.random.normal(command.p1, command.p2,len(ind))
            elif command.noise == "u":
                rands = np.random.uniform(command.p1, command.p2,len(ind))
            elif command.noise == None:
                rands = 0
            # TODO currently dep value gets overwritten!! --> No multicollinearities makeable
            self.dataset[dep] = command.raiser + command.slope * f + rands
        else:
            print("ERROR: At least one correlation partner is not in featurepool.")


    def add(self, *args):
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
        self.chainInfo = []
        self.relationChain.clear()
        self.outfilename = ""

    def printRelations(self):
        print("-"*30)
        print("Correlation chain to process:")
        count = 1
        for rel in self.relationChain:
            print("\t{0}:".format(count), rel.getInfo())
            count += 1

    def processRelations(self):
        for c in self.relationChain:
            self.chainInfo.append(c.getInfo())
            self.outfilename = "_".join(self.chainInfo)
            print("correlating: ", c.getInfo())
            self.correlate(c.dependent, c.independent, c.command)
        
    def data(self):
        return self.dataset

    def corTable(self):
        print("-"*30, "\nCorrelation Table")
        print(self.data().corr())

    def saveFile(self, topath, suffix=""):
        """Saves internally processed correlation file to given path.
        By default, correlations with the same parameters will get overwritten.
        Use suffix to use custom suffix which gets appended to the end of file."""
        if os.path.isdir(str(topath)):
            name = "[{m}x{n}]_".format(m=len(self.data()), n=len(self.data().columns))
            name += "_".join(self.chainInfo)
            fullname = topath + name + str(suffix) + ".csv"
            self.data().to_csv(fullname)
            print("saved to:", fullname)
            print("filesize:", round(os.path.getsize(fullname) / 1000000, 2), "MB.")
        else:
            print("ERROR: Can't save file. {op} does not exist.".format(op=topath))
        

#################################################
class Command:
    
    def __init__(self, slope=0, ftype="lin", noise=None, p1=0, p2=1, raiser=0, shift=0, power=1):
        self.ftypes = set(["lin", "exp", "sin", "cos", "log10", "log2", "pow"])
        # TODO make dist list
        self.noises = set(["n","u"])
        self.info = "default"
        self.make(slope, ftype, noise, p1, p2, raiser, shift, power)        
        # TODO make command id

    def make(self, slope=0, ftype="lin", noise=None, p1=0, p2=1, raiser=0, shift=0, power=1):
        # y = raiser + slope * f(x + shift)^power + noise(p1, p2)
        # ftype = ftype.lower()

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
        if noise == "n" or "u":
            self.noise = noise
            self.p1 = p1
            self.p2 = p2
        self.updateInfo()

    def updateInfo(self):
        # slope=0, ftype="lin", noise="none", p1=0, p2=1, raiser=0, shift=0, power=1):
        # (E<--y = 0 + 0.9 * lin(x + 0)^1 + normal(0,1) ---B)
        # self.info = 'y = {0} + {1} * {2}(x + {3})^{4} + {5}({6},{7}) '.format(
        #         self.raiser, self.slope, self.ftype, self.shift, self.power, self.noise, self.mu, self.sig)
        # self.info = '([{raiser}]+[{slope},{ftype},{shift},{power}]+[{noise},{p1},{p2}])'.format(
        # self.info = '-r{raiser},sl{slope},ft{ftype},sh{shift},p{power},no{noise},p1{p1},p2{p2}-'.format(
        self.info = '-{slope},{ftype},{noise}-{p1}-{p2}-'.format(
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
        return "[" + self.dependent + self.command.info + self.independent + "]"


########################################
def runTest():
    m = 30    # number of samples
    n = 4     # number of features: 4 --> [A,...,D]

    # make uncorrelated, initial data 
    cor = Correlator(m, n)

    # show head of initial data
    print("Raw random data:")
    print(cor.data().head())

    # make a chain of correlation commands in the following manner:
    # [(depFeature, indFeature, CorrCommand), (...), ...]
    #
    # Note that in the current implementation the dependent feature 
    # will be overwritten.
    # That means, one has to watch out for how to chain the commands
    # because otherways the chain might be broke unintentionally.
    # 
    # Correlation Commands specify the functional way, in which two
    # features get correlated. Use the follwing pattern:...
    # Command(slope=0, ftype="lin", noise="none", p1=0, p2=1, raiser=0, shift=0, power=1) 
    # ... to control the following functional relation between y (dep) and x (ind):
    # y = raiser + (slope * ftype(x + shift)^power + noise(p1, p2)
    # 
    # Currently, the following functional types and noise-distributions are supported:
    #   - ftype: "lin", "exp", "sin", "cos", "log10", "log2", "pow"
    #   - noise: "n", "u"
    # If "n" noise is selected p1 and p2 refer to mean and standard deviation.
    # If "u" noise is selected p1 and p2 refer to lower and upper distribution-bound.
    # 
    # First commands in the chain will be processed first.
    chain = [
        ("C","D", Command(0.9,"lin","n")),
        # ("A","C", Command(0.6,"lin","u",0,.4)),
    ]

    cor.add(*chain)
    cor.printRelations()
    cor.processRelations()

    # print("-"*30)
    # print("Data after correlation chain is processed:")
    # print(cor.data().head())

    # print pearson correlation table between all features
    cor.corTable()

    cor.saveFile("Data/")
    # cor.saveXml("Data/")
    
    data = cor.data()

    # vis(data["C"], data["A"], cor.relationChain[0].getInfo())
    # plt.plot(data, "x")
    # plt.show()


# runTest()





