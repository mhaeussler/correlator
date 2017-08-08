# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:50:52 2017

@author: Mo
"""

import matplotlib.pyplot as plt
import numpy as np


def command(raiser, slope, noise, p1, p2, shift=0, ftype="lin", power=1):
    """Command builder for convenience.
    
    Returns:
        command - dict.
    """

    ftypes = set(["lin", "exp", "sin", "cos", "log10", "log2", "pow"])
        
    if ftype not in ftypes:
        print("ERROR: funtion type not known. Assume linear...")
        ftype = ""
    if ftype == "lin":
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
        "power": power,
                
        }
    
    # (0, 1, "normal", 0, 1, 0, "lin")
    # 
    info = 'y = {0} + {1} * {2}(x + {3})^{4} + {5}({6},{7}) '.format(
            com["raiser"], com["slope"], com["ftype"], com["shift"], com["power"], com["noise"], p1, p2)
    
    return com, info



def make(x, com):
    """Generates y values for given x array and command.
    
    Returns:
        y - array."""
        
    # make function type
    ftype = com["ftype"].lower()
    k = com["shift"]
    f = 0
    
    if ftype == "":
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


def vis(x, y, info):
    xlimits = [-2,9]
    ylimits = [-3,14]
    plt.grid()
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.plot(x,y,"bx", label=info)
    plt.legend()
    plt.show()


########################################################
x = np.arange(-2, 20 , 0.3)
com, info = command(0, 1, "normal", 0, 1, 3, "pow")
y = make(x , com)

vis(x,y, info)
































