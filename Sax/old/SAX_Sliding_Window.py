# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:28:05 2018

@author: Rakesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from saxpy.hotsax import find_discords_hotsax
from saxpy.paa import paa
from saxpy.sax import sax_via_window
from saxpy.sax import sax_by_chunking
from sklearn.preprocessing import normalize
from saxpy.distance import euclidean
from sklearn import preprocessing
import math
import os


data2 =  pd.read_csv('car_sales.csv', sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten() 
#x1=np.array([x1])
x1=np.asfarray(x1,float)
#x1 = float(x1)






#x1 = x1 / np.linalg.norm(x1)
#x1 = (x1-min(x1))/(max(x1)-min(x1))
#normalized_X = preprocessing.normalize(x1)


plt.plot(x1)
plt.show()


window_size=10
word_size=3
alphabet_size=3
nr_strategy="exact"
z_threshold=0.01


#sax_via_window(data, win_size(segmentation), paa_size(word_size), alphabet_size,nr_strategy='exact', z_threshold=0.01):

sax_window = sax_via_window(x1, window_size, word_size, alphabet_size,nr_strategy, z_threshold)

sax_keys =list(sax_window.keys())
sax_values =list(sax_window.values())

elements_num = 0
for key in sax_window:
   elements_num += len(sax_window[key])
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
    plt.plot(x1)
    #plt.axvline(x=)
    plt.axvline(n1_val,c='r')
    #plt.plot(x1)
    plt.axvline(n1_val+2,c='r')
    plt.savefig('./Output/sliding_window/' +keyy+'.png')
    plt.show()    
        
       
#end of for loop
    
    
 
  
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
os.chdir('./Output/sliding_window/')
file = 'sax_via_sliding_window.xlsx'
writer = pd.ExcelWriter(file, engine='xlsxwriter')

df_all_sax=pd.DataFrame([sax_window])
df_all_sax.to_excel(writer, 'sax')
df_sax.to_excel(writer, 'sax_extended')
df_eucl_dist.to_excel(writer, 'euclidean_dist')
max_motif.to_excel(writer, 'max_motif')
discords.to_excel(writer, 'top_discords')
