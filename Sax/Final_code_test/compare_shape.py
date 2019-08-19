from collections import defaultdict
import itertools
from helper_functions import hamming_distance



def compare_shape_algo(seg_alpha,seg_indices,seg_df,word_length,ham_distance):
    """-------- Similar Words ------------- """
    sax = defaultdict(list)
    for i in range(0,len(seg_alpha)):
        if(len(seg_alpha[i])==word_length):
            sax[seg_alpha[i]].append(seg_indices[i])
    
    
    """-------- Compare Shape ------------- """
    
    map_keys = defaultdict(list)
    map_indices = defaultdict(list)
    simillar_word = sax
    for key_i in simillar_word:
        temp_list=list()
        temp_list.append(simillar_word.get(key_i))
        map_keys[key_i].append(key_i)
        
        for key_j in simillar_word:
            dist = hamming_distance(key_i, key_j)
            if(dist == ham_distance and key_i !=key_j):
                map_keys[key_i].append(key_j)
                temp_list.append(simillar_word.get(key_j))
            else:
                map_keys[key_i].append([])

        tempp = list(itertools.chain(*temp_list))
        map_indices[key_i].append(tempp)
        
        
    return (map_keys,map_indices)
