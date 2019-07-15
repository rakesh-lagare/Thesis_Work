import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import itertools


from helper_functions import normalize,alphabetize_ts,hamming_distance



"""-------------     Intialization     ------------- """
y_alphabet_size=4
word_lenth=3
window_size=10
skip_offset=5
ham_distance=0
epsilon = 1e-6

data =  pd.read_csv('ecg.csv', sep=',', header=None)
x1 = data.iloc[1:,1].values.flatten() 
x1=np.asfarray(x1,float)


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
        curr_count=curr_count+skip_offset-1

        temp_list=[]
        temp_list.append(sub_section)
        temp_df = pd.DataFrame(temp_list)
        temp_df.insert(loc=0, column='keys', value=curr_word)
        temp_df.insert(loc=1, column='position', value=sorted(sub_section)[len(sub_section) // 2])
        temp_df.insert(loc=2, column='scale_high', value=np.max(sub_section))
        temp_df.insert(loc=3, column='scale_low', value=np.min(sub_section))

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
        for key_j in simillar_word:
            dist=hamming_distance(key_i, key_j)
            if(dist==ham_distance and key_i !=key_j):
                map_keys[key_i].append(key_j)
                temp_list.append(simillar_word.get(key_j))
        tempp=list()
        tempp = list(itertools.chain(*temp_list))
        map_indices[key_i].append(tempp)        
    return (map_keys,map_indices)



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
                dist= np.linalg.norm(row1-row2)
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
            print(n1_val)
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
        
        
saxxx=complete_word()

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
        
prep_visualize ()
prep_visualize1 ()