# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:16:12 2019

@author: Meagatron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import itertools
from dtw import dtw
import timeit

from helper_functions import normalize,alphabetize_ts,hamming_distance



"""-------------     Intialization     ------------- """
start = timeit.default_timer()

data =  pd.read_csv('test_data2.csv', sep=',', header=None)
x1 = data.iloc[1:,1].values.flatten() 
x1=np.asfarray(x1,float)





y_alphabet_size=4
word_lenth=3
window_size=round( len(x1) *10 /100  )
skip_offset=round(window_size/2)
ham_distance=1
epsilon = 1e-6


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
        sub_section=normalize(sub_section)
        
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
        
   
        temp_df = pd.DataFrame()
        temp_df.insert(loc=0, column='sub_section', value=temp_list)
        temp_df.insert(loc=0, column='keys', value=curr_word)
        temp_df.insert(loc=0, column='position', value=sorted(sub_section)[len(sub_section) // 2])
        temp_df.insert(loc=0, column='scale_high', value=np.max(sub_section))
        temp_df.insert(loc=0, column='scale_low', value=np.min(sub_section))
        temp_df.insert(loc=0, column='indices', value=curr_count)
        
        
        curr_count=curr_count+skip_offset-1

        if(i==0):

            df_sax =temp_df.copy()
        else:
            df_sax=df_sax.append(temp_df, ignore_index=True)

    return (words,indices,df_sax)


alphabetize,indices,df_sax=segment_ts()



"""  Complete Words  """
def complete_word():
    
    complete_word=list()
    complete_indices=indices

    """  Simillar Words  """
    complete_word=alphabetize
    sax = defaultdict(list)
    for i in range(0,len(complete_word)):
        if(len(complete_word[i])==word_lenth):
            sax[complete_word[i]].append(complete_indices[i])
    return sax

#alphabetize1,indices1,df_sax=segment_ts()


def Compare_Shape():
    simillar_word=complete_word()
    map_keys = defaultdict(list)
    map_indices=defaultdict(list)
    
    
    for key_i in simillar_word:
        temp_list=list()
        temp_list.append(simillar_word.get(key_i))
        map_keys[key_i].append(key_i)
        
        for key_j in simillar_word:
            dist=hamming_distance(key_i, key_j)
            if(dist==ham_distance and key_i !=key_j):
                map_keys[key_i].append(key_j)
                temp_list.append(simillar_word.get(key_j))
            else:
                map_keys[key_i].append([])

        tempp = list(itertools.chain(*temp_list))
        map_indices[key_i].append(tempp)        
    return (map_keys,map_indices)




compare_strings,compare_list=Compare_Shape()


def  dtw_test2 ():
    df_dtw_prep=df_sax
    
    dtw_df=pd.DataFrame()
    
    
    for k, v in compare_list.items():
        
        v_temp=str(v)[2:-2]
        v1=[int(s) for s in v_temp.split(',')]

        for i in range(0,len(v1)-1):
            for j in range(i,len(v1)):
                
                
                if(v1[i] != v1[j]):
                    
                    

                    row1 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[i]]
                    row2 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[j]]
                    
                    sub_section1 = row1.iloc[0]['sub_section']
                    sub_section2 = row2.iloc[0]['sub_section']
                    
                    
                    index1 = row1.iloc[0]['indices']
                    index2 = row2.iloc[0]['indices']
                    

                    x=np.array(sub_section1).reshape(-1, 1)
                    y=np.array(sub_section2).reshape(-1, 1)

                    euclidean_norm = lambda x, y: np.abs(x - y)
                    dtw_value, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
                    
                    
                    temp_df = pd.DataFrame([[k,index1,index2,sub_section1,sub_section2,dtw_value]], 
                                           columns=['keyy','index1','index2','sub_section1','sub_section2','dtw_value'])
                    dtw_df=dtw_df.append(temp_df,ignore_index=True)
                    
    
    return(dtw_df)


dt_test=dtw_test2 ()


stop = timeit.default_timer()
print('Time: ', stop - start)  




















    



