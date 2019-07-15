import numpy as np
import pandas as pd
from segmentation import segment_ts
from compare_shape import Compare_Shape


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



def  matrix_prep (series,window_size,skip_offset,word_lenth,ham_distance):
    alphabetize,indices,feat_vector=segment_ts(series,window_size,skip_offset,word_lenth)
    compare_keys,compare_indices = Compare_Shape(series,window_size,skip_offset,word_lenth,ham_distance)
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