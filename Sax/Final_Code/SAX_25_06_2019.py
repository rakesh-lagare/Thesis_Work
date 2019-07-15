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
start=10
end=18
window_size=end-start
skip_offset=2#int(window_size/2)
y_alphabet_size=4
word_lenth=3
ham_distance=1
epsilon = 1e-6


"""-------------     import Data     -------------"""


data =  pd.read_csv('ecg.csv', sep=',', header=None)
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
xx2=x1[start:end]
plt.plot(xx2)
plt.show()

"""-------------   5.2-  Y_Alphabetize      ------------- """    
def alphabetize_ts(sub_section):
    mean_val=x_distrubted_values(sub_section)
    y_alpha_val=min(y_alphabets, key=lambda x:abs(x-mean_val))
    y_alpha_idx=y_alphabets.index(y_alpha_val)
    curr_word = index_to_letter(y_alpha_idx)

    return(curr_word)


"""-------------    2- Segmentization  Data      ------------- """    
def segment_ts():
    ts_len=len(x1)
    
    mod = ts_len%window_size
    
    rnge=0
    if(skip_offset==0):
     ts_len=int((ts_len-mod-window_size)/1)
     rnge=int(ts_len/window_size)
    else:
     ts_len=int(math.ceil((ts_len-window_size)/skip_offset))
     rnge=int(ts_len)
     
    curr_count=0
    words=list()
    indices=list()
    


    complete_indices=list()
    for i in range(0, rnge):
       
        sub_section = x1[curr_count:(curr_count+window_size)]
        #sub_section=normalize(sub_section)
        
        curr_word=""
        chunk_size=int(len(sub_section)/word_lenth)
        num=0
        curr_letter=""
        for j in range(0,word_lenth):
            chunk = sub_section[num:num + chunk_size]
            curr_letter=alphabetize_ts(chunk)
            curr_word+=str(curr_letter)
            complete_indices.append(curr_count)
            num+=chunk_size
           
        words.append(curr_word)
        indices.append(curr_count)
        
        
        temp_list=[]
        temp_list.append(sub_section)
        temp_df = pd.DataFrame(temp_list)
        temp_df.insert(loc=0, column='keys', value=curr_word)
        temp_df.insert(loc=1, column='position', value=sorted(sub_section)[len(sub_section) // 2])
        temp_df.insert(loc=2, column='scale_high', value=np.max(sub_section))
        temp_df.insert(loc=3, column='scale_low', value=np.min(sub_section))
        temp_df.insert(loc=4, column='idx', value=curr_count)
        
        curr_count=curr_count+skip_offset

        if(i==0):   
           
            df_sax =temp_df.copy()
        else:
            df_sax=df_sax.append(temp_df, ignore_index=True)

    return (words,indices,df_sax)




"""-------------    SAX      ------------- """    
alphabetize,indices,df_sax=segment_ts()

"""  Complete Words  """
def complete_word(series=x1,word_len=word_lenth,skip_len=skip_offset):
    alphabetize,indices,df_sax=segment_ts()
    complete_word=list()
    complete_indices=indices
    
    """  Simillar Words  """
    complete_word=alphabetize
    sax = defaultdict(list)
    for i in range(0,len(complete_word)):
        if(len(complete_word[i])==word_lenth):
            sax[complete_word[i]].append(complete_indices[i])
    return sax

simillar_word=complete_word()



#"""               CHANGE                        """

def Compare_Shape_Segments():
    simillar_word=complete_word()
    map_keys = defaultdict(list)
    map_indices=defaultdict(list)
    for key_i in simillar_word:
        temp_list=list()
        temp_list.append(simillar_word.get(key_i))
        for key_j in simillar_word:
            dist=hamming_distance(key_i, key_j)
            #print(dist)
            if(dist==ham_distance and key_i !=key_j):
                map_keys[key_i].append(key_j)
                #print(key_i)
                temp_list.append(simillar_word.get(key_j))
        tempp=list()
        tempp = list(itertools.chain(*temp_list))
        map_indices[key_i].append(tempp)
    print(map_keys.keys())
    print(map_keys.values())
    sax_keys =list(map_indices.keys())
    sax_values =list(map_indices.values())
    sax = defaultdict(list)
    for i in range(len(sax_values)):
        key=sax_keys[i]
        x2= list();
        for n_val in sax_values[i][0]:
            x2.append(n_val)
        sax[key] = x2
    
    
    #for key in map_keys.keys():
    
    
    
    
    
    
    
    
    return sax  












"""-------------    Selected- Segmentization  Data      ------------- """    
def selected_segment_ts():
    sub_section=xx2
    num=0
    alpha=""
    words=list()
    indices=list()
    curr_word=""
    chunk_size=int(len(sub_section)/word_lenth)
    #sub_section=normalize(sub_section)
    for j in range(0,word_lenth):
            chunk = sub_section[num:num + chunk_size]
            curr_word=alphabetize_ts(chunk)
            alpha+=str(curr_word)
            num+=chunk_size
    words.append(alpha)
    indices.append(start)
       
    return (words,indices)



"""  Complete Words  """
def selected_complete_word():
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





def Compare_Selected_Segments():
    simillar_word=Compare_Shape_Segments()   # CHANGE 
    selected_simillar_word=selected_complete_word()
    
    simillar_segs = {key:simillar_word[key] for key in selected_simillar_word if key in simillar_word}
    print(simillar_segs)
    
    return simillar_segs



def visualize(data,alph_size,lent,key):

    print(key)
    
    for i in range(0,lent):
        slice_range=slice(i*alph_size,(i+1)*alph_size)
        print(data)
        nData=data[slice_range]
        plt.plot(nData)
        pic_tag=str(i)
        plt.savefig('./ops/'+pic_tag+'.png')
        plt.show()



def  prep_visualize ():
    i=0
    simillar_word=Compare_Selected_Segments()
    sax_keys =list(simillar_word.keys())
    sax_values =list(simillar_word.values())
    
    
    for n_val in sax_values:
        #print(n_val)
        key=sax_keys[i]
        
        for n1_val in n_val:
            x2= list();
            alpha_count=0
            print(n1_val)
            while (alpha_count < window_size):
                x2.append(x1[n1_val+alpha_count])
                alpha_count=alpha_count+1
                
            
            plt.plot(x2)
            #plt.xlim([5, 18])
            #plt.xticks(n1_val, n1_val+alpha_count )
            plt.show()
        i=i+1

prep_visualize ()
    

