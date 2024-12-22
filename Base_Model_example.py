# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 03:46:52 2022

@author: yanbw
"""
import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import timeit
from sklearn import linear_model as lm
import csv
from modelBASE import TTCellModel


Ns = 1000

hypox=cp.Uniform(0,0.01)
hyper=cp.Uniform(0,0.01)
acid=cp.Uniform(0,0.01)

dist = cp.J(hypox,hyper,acid)
samples = dist.sample(Ns)

    
    
start = timeit.default_timer()


TTCellModel.run(samples.T,use_gpu=(False))
    
stop = timeit.default_timer()
time=stop-start
print('Time GPU: ',time) 


    
start = timeit.default_timer()

TTCellModel.run(samples.T,use_gpu=(True))
    
stop = timeit.default_timer()
time=stop-start
print('Time CPU: ',time) 

