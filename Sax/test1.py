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
start=20
end=42
window_size=end-start
skip_offset=int(window_size/2)





y_alphabet_size=4
word_lenth=3
#window_size=10
#skip_offset=5
ham_distance=0
epsilon = 1e-6




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

xx2=x1[start:end]
print(xx2)
xx2=normalize(xx2)

"""-------------   5.2-  Y_Alphabetize      ------------- """    
def alphabetize_ts(sub_section):
    mean_val=x_distrubted_values(sub_section)
    y_alpha_val=min(y_alphabets, key=lambda x:abs(x-mean_val))
    y_alpha_idx=y_alphabets.index(y_alpha_val)
    curr_word = index_to_letter(y_alpha_idx)
    return(curr_word)




"""-------------    2- Segmentization  Data      ------------- """    
def selected_segment_ts():
    sub_section=xx2
    num=0
    alpha=""
    words=list()
    indices=list()
    curr_word=""
    chunk_size=int(len(sub_section)/word_lenth)
    for j in range(0,word_lenth):
            chunk = sub_section[num:num + chunk_size]
            curr_word=alphabetize_ts(chunk)
            alpha+=str(curr_word)
            num+=chunk_size
    words.append(alpha)
    indices.append(start)
       
    return (words,indices)

segment=selected_segment_ts()

"""  Complete Words  """
def selected_complete_word(series=xx2):
    alphabetize,indices=selected_segment_ts()
    complete_word=list()
    complete_indices=indices
    
    """  Simillar Words  """
    complete_word=alphabetize
    sax = defaultdict(list)
    for i in range(0,len(complete_word)):
        if(len(complete_word[i])==word_lenth):
            sax[complete_word[i]].append(complete_indices[i])
    return sax

simillar_word=selected_complete_word()






