# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 19:56:15 2018

@author: Rakesh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import ts_to_string
from saxpy.hotsax import find_discords_hotsax
import math
import os


def sax_via_batch(series, win_size, paa_size, alphabet_size=3,
                   nr_strategy='none', z_threshold=0.01):
    """Simple via window conversion implementation."""
    cuts = cuts_for_asize(alphabet_size)
    sax = defaultdict(list)


    curr_count=0;
    next_count=win_size;
    cnt= len(series)//win_size
    print(cnt)
    
    for i in range(0, cnt):
        
        #sub_section = series[i:(i+win_size)]
        sub_section = series[curr_count:next_count]
        print(sub_section)
        zn = znorm(sub_section, z_threshold)

        paa_rep = paa(zn, paa_size)

        curr_word = ts_to_string(paa_rep, cuts)


        sax[curr_word].append(i)
        
        curr_count=next_count
        next_count=next_count+win_size

    return sax


window_size=3
word_size=3
alphabet_size=3
nr_strategy="exact"
z_threshold=0.01

data2 =  pd.read_csv('car_sales.csv', sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten()  # pick a random sample from class 0
x1 = x1.astype(np.float)

#sax_via_window(data, win_size(segmentation), paa_size(word_size), alphabet_size,nr_strategy='exact', z_threshold=0.01):

sax_batch = sax_via_batch(x1, window_size, word_size, alphabet_size,nr_strategy, z_threshold)

sax_keys =list(sax_batch.keys())
sax_values =list(sax_batch.values())

elements_num = 0
for key in sax_batch:
   elements_num += len(sax_batch[key])
elements_num


def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs



    
i=0
for n_val in sax_values:
    #print(x1[n_val])
    print(n_val)
    keyy=sax_keys[i]
    print(keyy)
    x2= list();
    
    for n1_val in n_val:
        #print(x1[n1_val], ",",x1[n1_val+1],",",x1[n1_val+2] )
        alpha_count=0
        while (alpha_count < alphabet_size):
            x2.append(x1[n1_val+alpha_count])
            alpha_count=alpha_count+1
   
    
    nn=split(x2,alphabet_size)
    df = pd.DataFrame(nn)
    df.insert(loc=0, column='key', value=keyy)
    df.insert(loc=1, column='start', value=n_val)
    #print("df",df)
    
    
    if(i==0):   
     df_sax =df.copy()
    else:
     df_sax=df_sax.append(df, ignore_index=True)
    

    
    i=i+1
    x3= list();
    for n2_val in x2:
        x3.append(n2_val)
    plt.plot(x3)
    plt.savefig('./Output/batch/' +keyy+'.png')
    plt.show() 
    
    
    
lenth= len(df_sax)
df_temp= df_sax.drop(columns=['key', 'start'])
for i in range(0,lenth-1):
    for j in (range( i+1,lenth)):
        key1=df_sax.iloc[i]['key']
        key2=df_sax.iloc[j]['key']
        if(key1==key2):
         row1= df_temp.iloc[[i]].values[0]
         row2= df_temp.iloc[[j]].values[0]
         eucl_dist= np.linalg.norm(row1-row2)
         eucl_dist= eucl_dist/math.sqrt((word_size))
         l1=([key1,row1,row2,eucl_dist])
         if(i==0 and j==1):
          df_eucl_dist= pd.DataFrame([l1])
         else:
          df_eucl_dist=df_eucl_dist.append([l1], ignore_index=True)
        

max_motif= df_eucl_dist.groupby(0).max()
sorted_motif= df_eucl_dist.sort_values(by=[3])

#find_discords_hotsax(series, win_size, num_discords, a_size,paa_size, z_threshold=0.01)
discords= find_discords_hotsax(x1, window_size, 5)
discords= pd.DataFrame(discords)

cwd = os.getcwd()
os.chdir("./Output/batch/")
file = 'sax_via_batch.xlsx'
writer = pd.ExcelWriter(file, engine='xlsxwriter')

df_all_sax=pd.DataFrame([sax_batch])
df_all_sax.to_excel(writer, 'sax')
df_sax.to_excel(writer, 'sax_extended')
df_eucl_dist.to_excel(writer, 'euclidean_dist')
max_motif.to_excel(writer, 'max_motif')
discords.to_excel(writer, 'top_discords')    