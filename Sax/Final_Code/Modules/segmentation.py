import numpy as np
import pandas as pd
import math
from helper_functions import normalize,alphabetize_ts



def segment_ts(series,window_size,skip_offset,word_lenth):


    ts_len=len(series)

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

        sub_section = series[curr_count:(curr_count+window_size)]
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
