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
import matplotlib



"""-------------     import Data     -------------"""
"""
file_name='test_data2.csv'
data2 =  pd.read_csv(file_name, sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten() 
x1 = x1.astype(np.float)
"""



data2 =  pd.read_csv('car_sales.csv', sep=',', header=None)

x1 = data2.iloc[1:,1].values.flatten() 
x1=np.asfarray(x1,float)







"""-------------     Intialization     ------------- """
y_alphabet_size=4
word_lenth=3
window_size=10
skip_offset=1


"""-------------     Helper Functions     ------------- """

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs




def distance_calculatation(df,key):
    width= len(df[0])
    mat = [[0 for x in range(width)] for y in range(width)]
    if(width>=3):
        for i in range(len(df)):
            for j in range(len(df)):
                row1= df.iloc[[i]].values[0]
                row2= df.iloc[[j]].values[0]
                eucl_dist= np.linalg.norm(row1-row2)
                eucl_dist= eucl_dist/math.sqrt((word_lenth))
                mat[i][j]=round(eucl_dist,2)
        print(mat)

"""

def visualize(start,alph_size,data ):
    plt.plot(x1)
    plt.plot(range(start,start+alph_size),data)
    plt.show()

"""



"""-------------     Y-axis Distribution      ------------- """
y_alphabets= np.linspace(0, 1, y_alphabet_size+1)[1:]
y_alphabets = y_alphabets.tolist()



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


alphabetize= segment_ts(x1)
len(alphabetize)

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


comp_word=complete_word()



"""-------------     Simlliar Words      ------------- """    
def simillar_words():
    simlliarWords= complete_word()
    sax = defaultdict(list)
    for i in range(0,len(simlliarWords)):
        if(len(simlliarWords[i])==word_lenth):
            sax[simlliarWords[i]].append(skip_offset*(i))
        
    return sax  



simillar_word=simillar_words()



def visualize(data,alph_size,lent ):
    row=int(lent/4)
    
    print(row)
    if(row > 1):
        fig = plt.figure(figsize=(4*row, 5*row))
        for i in range(0,lent):
            ax = fig.add_subplot(row+1, 4,i+1 )
            #plt.plot(x1)
            nData=data[i*alph_size:((i+1)*alph_size)]
            plt.plot(nData)
    else:
        fig = plt.figure(figsize=(4*3, 5*3))
        for i in range(0,lent):
            ax = fig.add_subplot(5, 2,i+1 )
            plt.plot(data)
    plt.show()   
     


   
    
"""-------------     Euclidean Distance      ------------- """ 
def  dist_matrix ():
    i=0
    per=int(len(x1)*25/100)
    #print(per)
    simillar_word=simillar_words()
    sax_keys =list(simillar_word.keys())
    sax_values =list(simillar_word.values())
    alphabet_size=window_size+(skip_offset*(word_lenth-1))
    for n_val in sax_values:
        keyy=sax_keys[i]
        #print(len(n_val ))
        
        x2= list();
        sum_alpha_list=list()
        for n1_val in n_val:
            #slice_range=slice(n1_val,n1_val+alphabet_size)
            #slice_data = x1[slice_range]
            #visualize(n1_val,alphabet_size,slice_data)
            alpha_count=0
            while (alpha_count < alphabet_size):
                x2.append(x1[n1_val+alpha_count])
                alpha_count=alpha_count+1
        visualize(x2,alphabet_size,len(n_val ))
        
        #print(x2)
        temp_list=split(x2,alphabet_size)
        temp_df = pd.DataFrame(temp_list)
        distance_calculatation(temp_df,keyy) 
    
        for j in range(0,len(temp_list)):
            sum_alphas=sum(temp_list[j])
            sum_alpha_list.append(sum_alphas)
            
        
        temp_df.insert(loc=0, column='key', value=keyy)
        temp_df.insert(loc=1, column='start', value=n_val)
        temp_df.insert(loc=2, column='sum', value=sum_alpha_list)
    
    
        if(i==0):   
            df_sax =temp_df.copy()
        else:
            df_sax=df_sax.append(temp_df, ignore_index=True)
            
        i=i+1
        
    return df_sax





sax=dist_matrix()
























































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
