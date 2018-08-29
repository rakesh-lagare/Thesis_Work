
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from saxpy.hotsax import find_discords_hotsax
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.sax import sax_via_window


data2 =  pd.read_csv('car_sales.csv', sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten()  # pick a random sample from class 0
x1 = x1.astype(np.float)

plt.plot(x1)
plt.show()


window_size=20
word_size=3
alphabet_size=3
nr_strategy="exact"
z_threshold=0.01



#sax_via_window(data, win_size(segmentation), paa_size(word_size), alphabet_size,nr_strategy='exact', z_threshold=0.01):

sax_window = sax_via_window(x1, window_size, word_size, alphabet_size,nr_strategy, z_threshold)

sax_keys =list(sax_window.keys())
sax_values =list(sax_window.values())


i=0
for n_val in sax_values:
    print(x1[n_val])
    print(n_val)
    print(sax_keys[i])
    x2= list();
    
    for n1_val in n_val:
        #print(x1[n1_val], ",",x1[n1_val+1],",",x1[n1_val+2] )
        alpha_count=0
        while (alpha_count < alphabet_size):
            x2.append(x1[n1_val+alpha_count])
            alpha_count=alpha_count+1
    x3= list();
    #print(x2)
    i=i+1
    for n2_val in x2:
        x3.append(n2_val)
    plt.plot(x3)
    plt.show()    
        
       



elements_num = 0
for key in sax_window:
   elements_num += len(sax_window[key])
elements_num







#find_discords_hotsax(series, win_size, num_discords, a_size,paa_size, z_threshold=0.01)
discords= find_discords_hotsax(x1, window_size, 5)