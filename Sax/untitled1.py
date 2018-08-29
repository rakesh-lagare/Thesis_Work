# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 18:25:37 2018

@author: Meagatron
"""
import os
import numpy as np
import math
import pandas as pd


s = SAX()

data2 =  pd.read_csv('car_sales.csv', sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten()  # pick a random sample from class 0
x1 = x1.astype(np.float)

#x1 = genfromtxt("car_sales.csv", delimiter=',')

(a,b)= s.to_PAA(x1)
print(a,b)

(lists)= s.sliding_window(x1,8,0.4)
# sliding_window(data , numSubsequences , overlappingFraction )
#the number of subsequences and how much each subsequence overlaps with the previous subsequence

#x1 = genfromtxt("ecg0606_1.csv", delimiter=',')
#(paa_data,paa_indices)= s.to_PAA(x1) 
#norm = s.normalize(x1)
#alphaSize = s.alphabetize(paa_data)



