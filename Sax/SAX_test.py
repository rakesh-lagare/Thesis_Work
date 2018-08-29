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


data2 =  pd.read_csv('car_sales.csv', sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten()  # pick a random sample from class 0
x1 = x1.astype(np.float)






#x1 = x1 / np.linalg.norm(x1)
x1 = (x1-min(x1))/(max(x1)-min(x1))
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


#mydf = pd.DataFrame(0,index=alphabet_size)
#mydf = pd.DataFrame(np.zeros((elements_num, alphabet_size)))
#mydf.append([3,4,5])
     
 
def Euclidean_Dist(df):
  lenth= len(df)
  num=0
  while (num < lenth):
    
    row_num= df.iloc[[num]].values[0]
    mm=df.apply(lambda row: np.linalg.norm(row-row_num)  , axis=1)
    num=num+1  
  return mm
    
    
i=0



for n_val in sax_values:
    print(x1[n_val])
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
             
    x3= list();
    
    nn=split(x2,3)
    
    df = pd.DataFrame(nn)
    df.insert(loc=0, column='key', value=keyy)
    print("df",df)
   # diff= Euclidean_Dist(df)
    #print("difffffff",diff)
    i=i+1
    for n2_val in x2:
        x3.append(n2_val)
    plt.plot(x3)
    plt.show()    
        
       








#find_discords_hotsax(series, win_size, num_discords, a_size,paa_size, z_threshold=0.01)
discords= find_discords_hotsax(x1, window_size, 5)