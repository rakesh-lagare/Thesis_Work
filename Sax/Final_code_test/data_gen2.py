# -*- coding: utf-8 -*-


from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nprnd
import pandas as pd
import os
os.remove("dataframe.csv")
os.remove("dataList.csv")


def pattern_gen(clas,noise,scale,offset):
    
    ts_data=[]
    tsn= [10,10,10,10,13,10,10,10,13,10,10,10,10,10,10,10,13,10,10,10,13,10,10,10]
    ts_noise = nprnd.randint(15, size=100)
    ts_n0ise = nprnd.randint(5, size=100)
    #box
    if (clas == 1):
        ts_data= [10,20,30 ,40,50,60,70,70,70,70,70,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,
                  80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,20,10]
        
        if(scale == 1):
            ts_data =[i * 2.5 for i in ts_data]
            
        if(noise == 1):
            
            ts_data = [sum(x) for x in zip(ts_data, ts_noise)]
        
    #linear increase
    elif (clas == 2):
        ts_data = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                   41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]
            
        if(scale == 1):
            ts_data =[i * 2.5 for i in ts_data]
            
        if(noise == 1):
            ts_data = [sum(x) for x in zip(ts_data, ts_noise)]
            
    #linear decrease
    elif (clas == 3):
        ts_data = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                   41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]
        ts_data.reverse()
            
        if(scale == 1):
            ts_data =[i * 2.5 for i in ts_data]

        if(noise == 1):
            ts_data = [sum(x) for x in zip(ts_data, ts_noise)]
            
    #periodic
    elif (clas == 4):
        ts_data = [20,30,40,50,60,70,80,90,70,60,50,40,30,20,10,10,10,10,10,20,30,40,50,60,70,80,90,70,60,50,
                   40,30,20,10,10,10,10,10,20,30,40,50,60,70,80,90,70,60,50,40,30,20]
        
        
        if(scale == 1):
            ts_data =[i * 3.5 for i in ts_data]
            
        if(noise == 1):
            ts_data = [sum(x) for x in zip(ts_data, ts_noise-10)]
            
            
            
    elif (clas == 5):
        ts_data = [20,30,85,88,90,88,85,36,34,36,55,60,58,20,20,18,18,20,20,90,85,55,55,55,60,
           10,20,30,85,88,90,88,85,36,34,36,55,60,58,20,20,18,18,20,20,90,85,55,55,55,60,10]
        
        
        if(scale == 1):
            ts_data =[i * 3.5 for i in ts_data]
            
        if(noise == 1):
            ts_data = [sum(x) for x in zip(ts_data, ts_noise)]
            
    elif (clas == 6):
        ts_data = [10,20,30,90,90,90,90,90,90,90,90,90,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,
             55,55,55,55,55,55,55,55,90,90,90,90,90,90,90,90,90,30,20,10]
        
        
        if(scale == 1):
            ts_data =[i * 3.5 for i in ts_data]
            
        if(noise == 1):
            ts_data = [sum(x) for x in zip(ts_data, ts_noise)]



    tss=  tsn + ts_data + tsn
    return (tss)    



def prep_data(num):
    df = pd.DataFrame()
    data_list = [] 
    for idx in range(num):
        
        
        random_clas = randrange(4) + 1
        random_noise = 1#randrange(2)
        random_scale = randrange(2)
        random_offset = 0#randrange(2)
        

        if(random_scale == 0):
            clas = random_clas + 0.1
        else:
            clas = random_clas + 0.2
        
        ts_data = pattern_gen(random_clas,random_noise,random_scale,random_offset)
        temp_df = pd.DataFrame([[idx,clas,random_noise,ts_data]], columns=['index','class','noise','data'])
        df = df.append(temp_df)

        data_list.extend(ts_data)

    plt.plot(data_list)
    
    plt.show()
    return (df,data_list)





df,data_list = prep_data(100)
























export_csv = df.to_csv (r'C:\Megatron\Thesis\Thesis_Work\Sax\Final_code_test\dataframe.csv', index = None, header=True)

temp_df = pd.DataFrame()
temp_df.insert(loc=0, column='sub_section', value=data_list)

export_csv1 = temp_df.to_csv (r'C:\Megatron\Thesis\Thesis_Work\Sax\Final_code_test\dataList.csv', index = None, header=True)

  