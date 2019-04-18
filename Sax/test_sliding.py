# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:51:11 2019

@author: Meagatron
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt,mpld3
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances    
from flask import Flask, render_template, request
import math
import itertools



"""-------------     Intialization     ------------- """
start=12
end=35
window_size=end-start
skip_offset=int(window_size/2)
y_alphabet_size=5
word_lenth=3
ham_distance=0
epsilon = 1e-6


"""-------------     import Data     -------------"""


data =  pd.read_csv('car_sales.csv', sep=',', header=None)
x1 = data.iloc[1:,1].values.flatten() 
x1=np.asfarray(x1,float)
#os.remove("./Output/sliding_half_segment/")

"""-------------     Helper Functions     ------------- """

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs


"""-------------     Y-axis Distribution      ------------- """
def break_points_gaussian(size):
    options = {
            3: np.array([ -0.43, 0.43]),
            4: np.array([ -0.67, 0, 0.67]),
            5: np.array([ -0.84, -0.25, 0.25, 0.84]),
            6: np.array([ -0.97, -0.43, 0, 0.43, 0.97]),
            7: np.array([ -1.07, -0.57, -0.18, 0.18, 0.57, 1.07]),
            8: np.array([ -1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15]),
            9: np.array([ -1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22]),
            10: np.array([ -1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]),
            11: np.array([ -1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34]),
            12: np.array([ -1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38]),
            13: np.array([ -1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43]),
            14: np.array([ -1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47]),
            15: np.array([ -1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5]),
            16: np.array([ -1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53]),
            17: np.array([ -1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56]),
            18: np.array([ -1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59]),
            19: np.array([ -1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62]),
            20: np.array([ -1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]),
            }

    return options[size]


def  break_points_quantiles(size):
    options=np.linspace(0, 1, size+1)[1:]
    return options


#y_alphabets = break_points_quantiles(y_alphabet_size).tolist()
y_alphabets = break_points_gaussian(y_alphabet_size).tolist()


def hamming_distance1(string1, string2): 
    distance = 0
    L = len(string1)
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

"""-------------     X-axis Distribution      ------------- """

def x_distrubted_values(series):
    mean=np.mean(series)
    #median=sorted(series)[len(series) // 2]
    return mean




"""-------------     Index to Letter conversion      ------------- """    

def index_to_letter(idx):
    """Convert a numerical index to a char."""
    if 0 <= idx < 20:
        return chr(97 + idx)
    else:
        raise ValueError('A wrong idx value supplied.')


def normalize(x):
        X = np.asanyarray(x)
        if np.nanstd(X) < epsilon:
            res = []
            for entry in X:
                if not np.isnan(entry):
                    res.append(0)
                else:
                    res.append(np.nan)
            return res
        return (X - np.nanmean(X)) / np.nanstd(X)

def normal_distribution(x):
    x = (x-min(x))/(max(x)-min(x))
    return x

"""-------------    1- Normalize Data      ------------- """    
x1=normalize(x1)

plt.plot(x1)
plt.show()

"""-------------   5.2-  Y_Alphabetize      ------------- """    
def alphabetize_ts(sub_section):

    mean_val=np.mean(sub_section)
    print(mean_val)
    y_alpha_val=min(y_alphabets, key=lambda x:abs(x-mean_val))
    #print("y_alpha_val",y_alpha_val)
    y_alpha_idx=y_alphabets.index(y_alpha_val)
    curr_word = index_to_letter(y_alpha_idx)
    #print(curr_word)
    return(curr_word)



"""-------------    2- Segmentization  Data      ------------- """    
def segment_ts():
    curr_count=0
    words=list()
    indices=list()

    complete_indices=list()
    for k in range(len(x1-window_size)):
        
        sub_section=x1[k : (k+window_size)]
        #sub_section=normalize(sub_section)
        #print(curr_count,(curr_count+window_size))
        #print(sub_section)
        
        curr_word=alphabetize_ts(sub_section)
        words.append(curr_word)
        k=k+skip_offset-1
        
        
        
        

    return (words)

ddd=segment_ts()
