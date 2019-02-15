# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:01:39 2019

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
y_alphabet_size=4
word_lenth=5
window_size=100
skip_offset=50
ham_distance=1
epsilon = 1e-6


"""-------------     import Data     -------------"""
"""
file_name='test_data2.csv'
data2 =  pd.read_csv(file_name, sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten() 
x1 = x1.astype(np.float)
"""

data =  pd.read_csv('test_data2.csv', sep=',', header=None)
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
    mean_val=x_distrubted_values(sub_section)
    y_alpha_val=min(y_alphabets, key=lambda x:abs(x-mean_val))
    y_alpha_idx=y_alphabets.index(y_alpha_val)
    curr_word = index_to_letter(y_alpha_idx)

    return(curr_word)



"""-------------    2- Segmentization  Data      ------------- """    
def segment_ts(series,windowSize=window_size,skip_offset=skip_offset):
    ts_len=len(x1)
    mod = ts_len%windowSize
    rnge=0
    if(skip_offset==0):
     ts_len=int((ts_len-mod-window_size)/1)
     rnge=int(ts_len/window_size)
    else:
     ts_len=int(math.ceil((ts_len-mod-window_size)/skip_offset))
     rnge=int(ts_len)
    curr_count=0
    words=list()
    indices=list()
    complete_indices=list()
    for i in range(0, rnge):
        sub_section = series[curr_count:(curr_count+windowSize)]
        sub_section=normalize(sub_section)
        #print(curr_count,(curr_count+windowSize))
        #print(sub_section)
        curr_word=alphabetize_ts(sub_section)
        chunk_size=int(len(sub_section)/word_lenth)
        num=0
        zlp=""
        for j in range(0,word_lenth):
            chunk = sub_section[num:num + chunk_size]
            curr_word=alphabetize_ts(chunk)
            zlp+=str(curr_word)
            complete_indices.append(curr_count)
            num+=chunk_size
        words.append(zlp)
        indices.append(curr_count)
        curr_count=curr_count+skip_offset-1
        
        temp_list=[]
        temp_list.append(sub_section)
        temp_df = pd.DataFrame(temp_list)
        temp_df.insert(loc=0, column='keys', value=zlp)
        temp_df.insert(loc=1, column='position', value=sorted(sub_section)[len(sub_section) // 2])
        temp_df.insert(loc=2, column='scale_high', value=np.max(sub_section))
        temp_df.insert(loc=3, column='scale_low', value=np.min(sub_section))

        if(i==0):   
           
            df_sax =temp_df.copy()
        else:
            df_sax=df_sax.append(temp_df, ignore_index=True)

    return (words,indices,df_sax)




"""-------------    SAX      ------------- """    


"""  Complete Words  """
def complete_word(series=x1,word_len=word_lenth,skip_len=skip_offset):
    alphabetize,indices,df_sax=segment_ts(series)
    complete_word=list()
    complete_indices=indices
    
    """
    for i in range(0, len(alphabetize)):
    	complete_word.append(alphabetize[i:i + word_len])
    n=0
    for i in complete_word:
        if(len(i) != word_len):
            complete_word.remove(i)
        n=n+1
         
                 
    """
    """  Simillar Words  """
    complete_word=alphabetize
    sax = defaultdict(list)
    for i in range(0,len(complete_word)):
        if(len(complete_word[i])==word_lenth):
            sax[complete_word[i]].append(complete_indices[i])
    return sax




simillar_word=complete_word()



"""-------------    Compare Shape Algorithm      ------------- """    


def Compare_Shape():
    simillar_word=complete_word()
    map_keys = defaultdict(list)
    map_indices=defaultdict(list)
    for key_i in simillar_word:
        temp_list=list()
        temp_list.append(simillar_word.get(key_i))
        for key_j in simillar_word:
            dist=hamming_distance(key_i, key_j)
            if(dist==ham_distance and key_i !=key_j): #
                map_keys[key_i].append(key_j)
                temp_list.append(simillar_word.get(key_j))
        tempp=list()
        tempp = list(itertools.chain(*temp_list))
        map_indices[key_i].append(tempp)        
    return (map_keys,map_indices)        


comapre=Compare_Shape()

"""-------------     Visualization      ------------- """  

def visualize(data,alph_size,lent,key):
    row=int(lent/4)
    print(key)
    if(lent > 4):
        fig = plt.figure(figsize=(4*row, 5*row))
        #ax.set_ylim(-2.5,2.5)
        for i in range(0,lent):
            slice_range=slice(i*alph_size,(i+1)*alph_size)
            nData=data[slice_range]
            fig.add_subplot(row+1, 4,i+1 )
            plt.plot(nData)
    else:
        fig = plt.figure(figsize=(3*3, 4*3))
        for i in range(0,lent):
            slice_range=slice(i*alph_size,(i+1)*alph_size)
            nData=data[slice_range]
            fig.add_subplot(5, 2,i+1 )
            plt.plot(nData)
    #plt.savefig('./Output/sliding_half_segment/'+key+'.png')
    #plt.savefig('books_read.png')        
    plt.show()
    

def  prep_visualize ():
    i=0
    simillar_word=complete_word()
    sax_keys =list(simillar_word.keys())
    sax_values =list(simillar_word.values())
    

    for n_val in sax_values:
        key=sax_keys[i]
        x2= list();
        for n1_val in n_val:
            alpha_count=0
            while (alpha_count < window_size):
                x2.append(x1[n1_val+alpha_count])
                alpha_count=alpha_count+1
            
        visualize(x2,window_size,len(n_val ),key)
        i=i+1

def  prep_visualize1 ():
    compare_keys,compare_indices = Compare_Shape()
    sax_keys =list(compare_indices.keys())
    sax_values =list(compare_indices.values())
    for i in range(len(sax_values)):
        key=sax_keys[i]
        x2= list();
        for n_val in sax_values[i][0]:
            alpha_count=0
            while (alpha_count < window_size):
                x2.append(x1[n_val+alpha_count])
                alpha_count=alpha_count+1
        visualize(x2,window_size,len(sax_values[i][0]),key)
    
    




"""-------------     Matrix      ------------- """  


def  matrix_calculation (df,key):
    df_temp = df.drop(columns=[ 'indexx','simillar_key'])
    width=len(df)
    s = (width,width)
    mat = np.zeros(s)
    
    if(width>=3):
        for i in range(len(df)):
            for j in range(len(df)):
                row1= df_temp.iloc[[i]].values[0]
                row2= df_temp.iloc[[j]].values[0]
                dist= row1-row2
                mat[i][j]=(dist) 

          
        dist_array = np.triu(mat, 0)
        print(key)
        print(dist_array)


def  matrix_prep ():
    alphabetize,indices,feat_vector=segment_ts(x1)
    compare_keys,compare_indices = Compare_Shape()
    sax_keys =list(compare_keys.keys())
    sax_values =list(compare_keys.values())
    
    i=0
    
    for n_val in sax_values:
        key=sax_keys[i]
        temp_df = pd.DataFrame()
        index_list=list()
        position_list=list()
        simillar_key_list=list()
        for n1_val in n_val:
            for index, row in feat_vector.iterrows():
                if(row['keys']==n1_val):
                    # print(row['position'],index)
                    index_list.append(index)
                    position_list.append(row['position'])
                    simillar_key_list.append(n1_val)
                    
        temp_df['indexx']=index_list
        temp_df['position']=position_list
        temp_df['simillar_key']=simillar_key_list            
        matrix_calculation(temp_df,key)            
                    
                    
        i=i+1

print("===========================Before Compare Shape============================")
prep_visualize()

print("===========================After Compare Shape============================")
prep_visualize1 ()

print("===========================Position Matrix ============================")
matrix_prep()


