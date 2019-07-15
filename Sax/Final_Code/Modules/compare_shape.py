from collections import defaultdict
import itertools
from helper_functions import hamming_distance
from segmentation import segment_ts



"""  Complete Words  """
def complete_word(series,window_size,skip_offset,word_lenth):
    alphabetize,indices,df_sax=segment_ts(series,window_size,skip_offset,word_lenth)
    complete_word=list()
    complete_indices=indices

    """  Simillar Words  """
    complete_word=alphabetize
    sax = defaultdict(list)
    for i in range(0,len(complete_word)):
        if(len(complete_word[i])==word_lenth):
            sax[complete_word[i]].append(complete_indices[i])
    return sax



def Compare_Shape(series,window_size,skip_offset,word_lenth,ham_distance):
    simillar_word=complete_word(series,window_size,skip_offset,word_lenth)
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