"""
Created on Thu Aug 28 16:28:33 2018

@author: Rakesh
"""

import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances    


"""-------------     import Data     -------------"""
"""
file_name='test_data2.csv'
data2 =  pd.read_csv(file_name, sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten() 
x1 = x1.astype(np.float)
"""



data =  pd.read_csv('car_sales.csv', sep=',', header=None)

x1 = data.iloc[1:,1].values.flatten() 
x1=np.asfarray(x1,float)




"""-------------     Intialization     ------------- """
y_alphabet_size=4
word_lenth=3
window_size=5
skip_offset=5


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
        2: np.array([-np.inf,  0.00]),
        3: np.array([-np.inf, -0.43, 0.43]),
        4: np.array([-np.inf, -0.67, 0, 0.67]),
        5: np.array([-np.inf, -0.84, -0.25, 0.25, 0.84]),
        6: np.array([-np.inf, -0.97, -0.43, 0, 0.43, 0.97]),
        7: np.array([-np.inf, -1.07, -0.57, -0.18, 0.18, 0.57, 1.07]),
        8: np.array([-np.inf, -1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15]),
        9: np.array([-np.inf, -1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22]),
        10: np.array([-np.inf, -1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]),
        11: np.array([-np.inf, -1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34]),
        12: np.array([-np.inf, -1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38]),
        13: np.array([-np.inf, -1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43]),
        14: np.array([-np.inf, -1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47]),
        15: np.array([-np.inf, -1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5]),
        16: np.array([-np.inf, -1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53]),
        17: np.array([-np.inf, -1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56]),
        18: np.array([-np.inf, -1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59]),
        19: np.array([-np.inf, -1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62]),
        20: np.array([-np.inf, -1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]),
    }

    return options[size]


def  break_points_quantiles(size):
    options=np.linspace(0, 1, size+1)[1:]
    return options


y_alphabets = break_points_quantiles(y_alphabet_size).tolist()
#y_alphabets = break_points_gaussian(y_alphabet_size).tolist()





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




"""-------------     Normalize Data      ------------- """    
x1 = (x1-min(x1))/(max(x1)-min(x1))
plt.plot(x1)
plt.show()



"""-------------     Segmentation      ------------- """    

def segment_ts(series,windowSize=window_size,skip_offset=skip_offset):
    ts_len=len(x1)
    mod = ts_len%windowSize
    ts_len=int((ts_len-mod-windowSize)/skip_offset)
    curr_count=0
    sax = defaultdict(list)
    alphas=list()
    alpha_idx=list()
    alp=""
    for i in range(0, ts_len):
        sub_section = series[curr_count:(curr_count+windowSize)]
        curr_count=curr_count+skip_offset
        curr_word=alphabetize_ts(sub_section) 
        alphas.extend(curr_word)
        alpha_idx.append(i)
        sax[curr_word].append(i)
        alp+=str(curr_word)

    return alp   
          

"""-------------     Alphabetize      ------------- """    

def alphabetize_ts(sub_section):
    mean_val=x_distrubted_values(sub_section)
    y_alpha_val=min(y_alphabets, key=lambda x:abs(x-mean_val))
    y_alpha_idx=y_alphabets.index(y_alpha_val)
    curr_word = index_to_letter(y_alpha_idx)
    
    return(curr_word)


"""-------------     Complete Words      ------------- """    
def complete_word(series=x1,word_len=word_lenth,skip_len=0):
    alphabetize= segment_ts(series)
    complete_word=list()
    for i in range(0, len(alphabetize)):
    	complete_word.append(alphabetize[i:i + word_len]) 
    for i in complete_word:
        if(len(i) != word_len):
            complete_word.remove(i)
    
    return complete_word




"""-------------     Simlliar Words      ------------- """    
def simillar_words():
    simlliarWords= complete_word()
    sax = defaultdict(list)
    for i in range(0,len(simlliarWords)):
        if(len(simlliarWords[i])==word_lenth):
            sax[simlliarWords[i]].append(skip_offset*(i))
        
    return sax  

simillar_word=simillar_words()



"""-------------     Visualization      ------------- """  
def visualize(data,alph_size,lent,key):
    row=int(lent/4)
    print(key)
    #print(len(data))
    #print(alph_size)
    if(lent > 4):
        fig = plt.figure(figsize=(4*row, 5*row))
        for i in range(0,lent):
            slice_range=slice(i*alph_size,(i+1)*alph_size)
            nData=data[slice_range]
            fig.add_subplot(row+1, 4,i+1 )
            plt.plot(nData)
    else:
        fig = plt.figure(figsize=(3*3, 4*3))
        for i in range(0,lent):
            print()
            slice_range=slice(i*alph_size,(i+1)*alph_size)
            nData=data[slice_range]
            fig.add_subplot(5, 2,i+1 )
            plt.plot(nData)
    plt.show()   
     

        
"""-------------     Euclidean Distance      ------------- """ 
def dist_matrix(df,key):
    df_temp = df.drop(columns=[ 'start'])
    width= len(df_temp[0])
    s = (width,width)
    mat = np.zeros(s)
    #strt=df.ix[1:2, 0:1]
    if(width>=3):
      for i in range(len(df)):
            for j in range(len(df)):
                row1= df_temp.iloc[[i]].values[0]
                row2= df_temp.iloc[[j]].values[0]
                eucl_dist= np.linalg.norm(row1-row2)
                eucl_dist= eucl_dist/math.sqrt((word_lenth))
                mat[i][j]=round(eucl_dist,2) 
              
      dist_array = np.triu(mat, 0)
      print(key)
      print(dist_array)



def  distance_calculation ():
    i=0
    simillar_word=simillar_words()
    sax_keys =list(simillar_word.keys())
    sax_values =list(simillar_word.values())
    alphabet_size=window_size+(skip_offset*(word_lenth-1))
    for n_val in sax_values:
        key=sax_keys[i]
        x2= list();
        #sum_alpha_list=list()
        for n1_val in n_val:
            
            alpha_count=0
            while (alpha_count < alphabet_size):
                x2.append(x1[n1_val+alpha_count])
                alpha_count=alpha_count+1
        visualize(x2,alphabet_size,len(n_val ),key)
        
        temp_list=split(x2,alphabet_size)
        temp_df = pd.DataFrame(temp_list)
        temp_df.insert(loc=0, column='start', value=n_val)
        dist_matrix(temp_df,key) 
        temp_df.insert(loc=1, column='key', value=key)
        
        
        if(i==0):   
            df_sax =temp_df.copy()
        else:
            df_sax=df_sax.append(temp_df, ignore_index=True)
            
        i=i+1
        
    return df_sax



sax=distance_calculation()
























































"""

df_temp= sax.drop(columns=['key', 'start','sum'])

for i in range(len(sax)-1):
    for j in range(i+1,len(sax)):
        mat=list()
        key1=sax.iloc[i]['key']
        key2=sax.iloc[j]['key']
        if(key1==key2):
         #print(key1,key2)
         row1= df_temp.iloc[[i]].values[0]
         row2= df_temp.iloc[[j]].values[0]
         #print(row1,row2)
         eucl_dist= np.linalg.norm(row1-row2)
         eucl_dist= eucl_dist/math.sqrt((word_lenth))
         #mat = eucl_dist


"""







































"""
sax_keys =list(simillar_word.keys())
sax_values =list(simillar_word.values())




i=0
alphabet_size=5*3 #alphabet size * word length
for n_val in sax_values:
    #print(x1[n_val])
    #print(n_val)
    keyy=sax_keys[i]
    #print(keyy)
    x2= list();
    sum_alpha_list=list()
    sum_alphas=0
    for n1_val in n_val:
        #print(x1[n1_val], ",",x1[n1_val+1],",",x1[n1_val+2] )
        alpha_count=0
        while (alpha_count < alphabet_size):
            x2.append(x1[n1_val+alpha_count])
            alpha_count=alpha_count+1
            
   
    
    nn=split(x2,alphabet_size)
    df = pd.DataFrame(nn)
    
    for j in range(0,len(nn)):
        sum_alphas=sum(nn[j])
        print(sum_alphas)
        sum_alpha_list.append(sum_alphas)
    #print(sum_alphas)
    df.insert(loc=0, column='key', value=keyy)
    df.insert(loc=1, column='start', value=n_val)
    df.insert(loc=2, column='sum', value=sum_alpha_list)
    #print("df",df)
    
    
    if(i==0):   
     df_sax =df.copy()
    else:
     df_sax=df_sax.append(df, ignore_index=True)
    

    
    i=i+1
    x3= list();
    for n2_val in x2:
        x3.append(n2_val)
        
    #plt.plot(x1)
    #plt.axvline(n1_val,c='r')
   
    
    
    #plt.axvline(n1_val+2,c='r')
    #plt.savefig('./Output/sliding_window/' +keyy+'.png')
    #plt.show()   
    
"""   
