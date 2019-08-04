import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import itertools
from dtw import dtw
import timeit
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from statistics import median

from helper_functions import normalize,alphabetize_ts,hamming_distance



"""-------------     Intialization     ------------- """
start = timeit.default_timer()

data =  pd.read_csv('car_sales.csv', sep=',', header=None)
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






"""  Complete Words  """
def complete_word():
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



def  dtw_test2 ():
    alphabetize,indices,df_dtw_prep=segment_ts()
    compare_strings,compare_list=Compare_Shape()
    dtw_df=pd.DataFrame()
    dtw_rank_df=pd.DataFrame()
    
    
    for k, v in compare_list.items():
        
        v_temp=str(v)[2:-2]
        v1=[int(s) for s in v_temp.split(',')]
        dtw_temp=pd.DataFrame()
        for i in range(0,len(v1)-1):
            for j in range(i,len(v1)):
                
                
                if(v1[i] != v1[j]):
                    
                    

                    row1 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[i]]
                    row2 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[j]]
                    
                    sub_section1 = row1.iloc[0]['sub_section']
                    sub_section2 = row2.iloc[0]['sub_section']
                    
                    
                    index1 = row1.iloc[0]['indices']
                    index2 = row2.iloc[0]['indices']
                    
                    """------------------Normal DTW-------------------------"""
                    #x=np.array(sub_section1).reshape(-1, 1)
                    #y=np.array(sub_section2).reshape(-1, 1)

                    #euclidean_norm = lambda x, y: np.abs(x - y)
                    #dtw_value, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
                    
                    
                    #temp_df = pd.DataFrame([[k,index1,index2,sub_section1,sub_section2,dtw_value]], columns=['keyy','index1','index2','sub_section1','sub_section2','dtw_value'])
                     
                     
                    """-----------------Fast DTW--------------------"""
                    x = np.array(sub_section1)
                    y = np.array(sub_section2)
                    dtw_value, path = fastdtw(x, y, dist=euclidean)
                    
                   #temp_df = pd.DataFrame([[k,index1,index2,sub_section1,sub_section2,dtw_value]], columns=['key','index1','index2','sub_section1','sub_section2','dtw_value'])
                    temp_df = pd.DataFrame([[k,index1,index2,sub_section1,sub_section2,dtw_value]], columns=['key','index1','index2','sub_section1','sub_section2','dtw_value'])
                    dtw_temp=dtw_temp.append(temp_df,ignore_index=True)
                    
                    
                    
                    """---------------------------------------------"""
                     
                     
                    #dtw_df=dtw_df.append(temp_df,ignore_index=True)

        med=(dtw_temp['dtw_value'] ).tolist()

        if(len(dtw_temp)> 5) :
            dtw_temp = dtw_temp[dtw_temp['dtw_value'] < median(med)]
        dtw_temp= dtw_temp.sort_values(by=['dtw_value'])
        rank_list=[]
        for m in range(1, len(dtw_temp)+1):
            rank_list.append(m)
            
        dtw_temp.insert(loc=6, column='ranks', value=rank_list)
        dtw_rank_df= dtw_rank_df.append(dtw_temp,ignore_index=True)
        

    return(dtw_rank_df)


dt_test=dtw_test2 ()
























stop = timeit.default_timer()
print('Time: ', stop - start)  

































def  matrix_calculation (df,key):
    df_temp = df.drop(columns=[ 'indexx','simillar_key'])
    width=len(df)
    s = (width,width)
    mat = np.zeros(s)
    #print(df_temp)
    if(width>=3):
        for i in range(len(df)):
            for j in range(len(df)):
                row1= df_temp.iloc[[i]].values[0]
                row2= df_temp.iloc[[j]].values[0]
                dist= np.linalg.norm(row1-row2)
                
                euclidean_norm = lambda x, y: np.abs(row1 - row2)

                d, cost_matrix, acc_cost_matrix, path = dtw(row1, row2, dist=euclidean_norm)
                
                mat[i][j]=(dist) 
                
        dist_array = np.triu(mat, 0)
        
        print(key)
        print(dist_array)



def  matrix_prep (series):
    
    alphabetize,indices,feat_vector=segment_ts()
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
                    index_list.append(row['indices'])
                    position_list.append(row['position'])
                    simillar_key_list.append(n1_val)

                    
        temp_df['indexx']=index_list
        temp_df['position']=position_list
        temp_df['simillar_key']=simillar_key_list            
        print(temp_df)
        
        matrix_calculation(temp_df,key)            
                    
                    
        i=i+1


#matrix_prep(x1)









    



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
        
#prep_visualize ()
#prep_visualize1 ()
        





"""-------------     DTW      ------------- """
"""------------------------------------------------------------------------------ """  
"""------------------------------------------------------------------------------ """
"""------------------------------------------------------------------------------ """  
"""------------------------------------------------------------------------------ """  
"""------------------------------------------------------------------------------ """      



def  dtw_prep ():
    
    alphabetize,indices,feat_vector=segment_ts()
    feat_vector = feat_vector [['indices','keys','sub_section']]
    compare_keys,compare_indices = Compare_Shape()
    temp_list1=[]
    temp_list=[]
    
    
    for index, row in feat_vector.iterrows():
        for k, v in compare_keys.items():
             if(str(row['keys'])==str(k)):
                 temp_list.append(v)
                 
                 
        for k1, v1 in compare_indices.items():
            if(str(row['keys'])==str(k1)):
                v1_temp1=str(v1)[2:-2]
                v1_temp2=[int(s) for s in v1_temp1.split(',')]
                temp_list1.append(v1_temp2)
                

    feat_vector["compare"]=temp_list
    feat_vector["compare_list"]=temp_list1
        
    return feat_vector


#testt=dtw_prep ()





def  dtw_prep_old (series):
    alphabetize,indices,df_sax=segment_ts()

    lenth= len(df_sax)
    
    #for index, row in df_sax.iterrows():
                #print(row)
    
    
    for i in range(0,lenth-1):
        for j in (range( i+1,lenth)):
            key1=df_sax.iloc[i]['keys']
            key2=df_sax.iloc[j]['keys']
            if(key1==key2):
                row11=df_sax.iloc[i]['sub_section']
                row22=df_sax.iloc[j]['sub_section']
                
                x=np.array(row11).reshape(-1, 1)
                y=np.array(row22).reshape(-1, 1)
                                
                euclidean_norm = lambda x, y: np.abs(x - y)

                d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
                print(df_sax.iloc[i]['keys'],"  ",df_sax.iloc[i]['indices'],"   " , df_sax.iloc[j]['indices'] )
                print(d)
    
    


def  dtw_prep_old2 (series):
    alphabetize,indices,df_sax=segment_ts()

    lenth= len(df_sax)
    

    
    for i in range(0,lenth-1):
        for j in (range( i+1,lenth)):
            key1=df_sax.iloc[i]['keys']
            key2=df_sax.iloc[j]['keys']
            if(key1==key2):
                row11=df_sax.iloc[i]['sub_section']
                row22=df_sax.iloc[j]['sub_section']
                
                x=np.array(row11).reshape(-1, 1)
                y=np.array(row22).reshape(-1, 1)
                                
                euclidean_norm = lambda x, y: np.abs(x - y)

                d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
                print(df_sax.iloc[i]['keys'],"  ",df_sax.iloc[i]['indices'],"   " , df_sax.iloc[j]['indices'] )
                print(d)

def  dtw_prep_old1 (series):
    alphabetize,indices,feat_vector=segment_ts()
    compare_keys,compare_indices = Compare_Shape()
    sax_keys =list(compare_indices.keys())
    sax_values =list(compare_indices.values())
    print(sax_keys)
    print(sax_values)
    
    i=0
    for n_val in sax_values:
        for n1_val in n_val:
            print(n1_val)
            temp_list=[]
            
            for n2_val in n1_val:
                rows= x1[n2_val:(n2_val+window_size)]
                temp_list.append(rows)

                #print(rows)
            #print(temp_list)
            temp_df = pd.DataFrame()
            temp_df.insert(loc=0, column='data', value=temp_list)
            temp_df.insert(loc=0, column='keys', value=sax_keys[i])
            temp_df.insert(loc=0, column='keyees', value=sax_keys[i])
            print(temp_df)
                
                
    i=i+1
    return(temp_df)
    
"""-------------     DTW      ------------- """
"""------------------------------------------------------------------------------ """  
"""------------------------------------------------------------------------------ """
"""------------------------------------------------------------------------------ """  
"""------------------------------------------------------------------------------ """  
"""------------------------------------------------------------------------------ """  
