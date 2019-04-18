import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#,mpld3
from collections import defaultdict
import math
import itertools


from pages.helper_functions import index_to_letter,x_distrubted_values
from pages.helper_functions import normalize,alphabetize_ts,hamming_distance



"""-------------     Intialization     ------------- """
y_alphabet_size=4
word_lenth=3
window_size=10
skip_offset=2
ham_distance=1
epsilon = 1e-6

start=404
end=404
seg_data=[404]
x1=[404]
x2=[404]





def get_segment_data(x_data,start_seg,end_seg):
    global seg_data,start,end,x1,window_size,skip_offset,x2


    seg_data=x_data
    start=int(start_seg)
    end=int(end_seg)
    print(start)
    print(end)

    x1=seg_data
    window_size=end-start
    skip_offset=2#int(window_size/2)


    x1=normalize(x1)
    x2=x1[start:end]

    plt.plot(x1)
    plt.show()
    plt.plot(x2)
    plt.show()

    #comp=selected_complete_word()
    #print(comp)
    prep_visualize ()






"""-------------    2- Segmentization  Data      ------------- """
def segment_ts():


    ts_len=len(x1)
    print("ts_len",ts_len)
    print("windowSize",window_size)
    print("skip_offset",skip_offset)



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
        #sub_section=normalize(sub_section)

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
        curr_count=curr_count+skip_offset

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


"""----------------------------------------------------------------------------------- """
"""-------------    Selected Segments      ------------- """
"""----------------------------------------------------------------------------------- """




def selected_segment_ts():
    sub_section=x2
    print(x2)
    num=0
    alpha=""
    words=list()
    indices=list()
    curr_word=""
    chunk_size=int(len(sub_section)/word_lenth)
    #sub_section=normalize(sub_section)
    for j in range(0,word_lenth):
            chunk = sub_section[num:num + chunk_size]
            curr_word=alphabetize_ts(chunk)
            alpha+=str(curr_word)
            num+=chunk_size
    words.append(alpha)
    indices.append(start)

    return (words,indices)



"""  Complete Words  """
def selected_complete_word():
    alphabetize,indices=selected_segment_ts()
    complete_word=list()
    complete_indices=indices

    print(alphabetize)
    print(indices)
    """  Simillar Words  """
    complete_word=alphabetize
    sax = defaultdict(list)
    for i in range(0,len(complete_word)):
        if(len(complete_word[i])==word_lenth):
            sax[complete_word[i]].append(complete_indices[i])

    return sax



def Compare_Selected_Segments():
    simillar_word=complete_word()
    selected_simillar_word=selected_complete_word()

    simillar_segs = {key:simillar_word[key] for key in selected_simillar_word if key in simillar_word}
    return simillar_segs

def visualize(data,alph_size,lent,key):

    print(key)
    path="C:/Megatron/Thesis/Thesis_Work/Django/Thesis/Motif_Discovery/Output/"

    for i in range(0,lent):
        slice_range=slice(i*alph_size,(i+1)*alph_size)
        nData=data[slice_range]
        plt.plot(nData)
        pic_tag=str(i)
        plt.savefig('./static/ops/'+pic_tag+'.png')
        plt.show()





def  prep_visualize ():
    i=0
    simillar_word=Compare_Selected_Segments()
    sax_keys =list(simillar_word.keys())
    sax_values =list(simillar_word.values())



    for n_val in sax_values:
        key=sax_keys[i]
        temp_x2= list();
        for n1_val in n_val:
            alpha_count=0
            while (alpha_count < window_size):
                temp_x2.append(x1[n1_val+alpha_count])
                alpha_count=alpha_count+1

        visualize(temp_x2,window_size,len(n_val ),key)
        i=i+1
