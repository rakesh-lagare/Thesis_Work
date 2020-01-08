import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import os
import glob
import math
from collections import defaultdict
from segmentation import segment_ts,dtw_segment_ts
from compare_shape import compare_shape_algo
from rank_table_with_params import rank_table_with_param
from rank_table_without_params import rank_table_without_param
from helper_functions import normalize
from evaluation import orinal_dtw_rank_tab,euclidean_rank_tab
from dtw_visualization import dtw_visualization_DTW





"""-------------     Intialization     ------------- """
start = timeit.default_timer()


""" ------------ OLD
pre_data =  pd.read_csv('data22.csv', sep=',', header=None)
pre_data = pre_data[:500000]
#pre_data = pre_data.fillna(0)
ts = pre_data.iloc[1:,1].values.flatten()

""" 

def remove_files():
    
    files1 = glob.glob('C:/Megatron/Thesis/Thesis_Work/Sax/Final_code_test/Output/with_param/*')
    for f in files1:
        os.remove(f)
        
    files2 = glob.glob('C:/Megatron/Thesis/Thesis_Work/Sax/Final_code_test/Output/without_param/*')
    for f in files2:
        os.remove(f)
        
    files3 = glob.glob('C:/Megatron/Thesis/Thesis_Work/Sax/Final_code_test/Output/DTW/*')
    for f in files3:
        os.remove(f)
    
    os.remove('C:/Megatron/Thesis/Thesis_Work/Sax/Final_code_test/Output/Original.png')
#remove_files()



#pre_data =  pd.read_csv('dataList.csv', sep=',', header=None)
#data_df = pd.read_csv('dataframe.csv', sep=',' )
#ts = pre_data.iloc[1:, :1]
    
    
pre_data =  pd.read_csv('ECG200.csv', sep=',', header=None)
ts = pre_data.iloc[:,0].values.flatten()



ts = np.asfarray(ts,float)
ts= normalize(ts)

plt.plot(ts)
plt.savefig('./Output/Original.png')
plt.show()

 
y_alpha_size = 4
word_length = 3
window_size = 120#round( len(ts) * 0.1 )
skip_offset = round(window_size   ) 
ham_distance = 0


seg_alpha, seg_indices,seg_df = segment_ts(ts,window_size,skip_offset,word_length,y_alpha_size)
#seg_dtw_df = dtw_segment_ts(ts,window_size,skip_offset)
compare_strings, compare_list = compare_shape_algo(seg_alpha,seg_indices,seg_df,word_length,ham_distance)



#print("--------------   Without Param  ----------------------------------")
tab_PA_class_rank_without = rank_table_without_param(seg_df, compare_list, window_size, ts)

#print("--------------   With Param  ----------------------------------")
#tab_PA_class_rank_with , temp_subClass_scale_with= rank_table_with_param(seg_df, compare_list, window_size, ts)

#print("--------------------------------Original DTW---------------------")
#tab_dtw  = orinal_dtw_rank_tab(seg_dtw_df)


#print("----------------------------Euclidean---------------------------")
#tab_euclidean = euclidean_rank_tab(seg_df)



stop = timeit.default_timer()
print('Time: ', stop - start)  



























""" ----------------------------  Evaluation -----------------------------------------"""

def sub_class_idx():
    print("Evaluation Started")
    temp_lst = []
    for k, v in temp_subClass_scale_with.items():

        unique_val = list(set(v))
        unique_val.sort()
        #print(unique_val)
        if(len(unique_val) > 1):
            for i in range(0,len(unique_val)-1):
                for j in range(i,len(unique_val)):
                    if(unique_val[i] != unique_val[j]):
                        indices=[unique_val[i],unique_val[j]]

                        temp_lst.append(indices)
        else:

            temp_lst.append(unique_val)
    

    return (temp_lst)


def PA_class_accuracy():
    map_data= defaultdict(list)
    for i in range(0,len(seg_df)-1):
        row1_idx = seg_df.iloc[i]['indices']
        row1_key = seg_df.iloc[i]['keys']

        for j in range(i,len(seg_df)):
            row2_idx = seg_df.iloc[j]['indices']
            row2_key = seg_df.iloc[j]['keys']
            if(row1_key == row2_key ):
                map_data[row1_key].append(row1_idx)
                map_data[row1_key].append(row2_idx)

    map_data_temp1 = defaultdict(list)
    for k,val in map_data.items():
        unique_val = list(set(val))
        map_data_temp1[k].append(unique_val)
        
    PA_with_param_class_list=[]
    for k,val in map_data_temp1.items():
        PA_with_param_class_list.append(val[0])
        
        
    PA_without_param_class_list = PA_with_param_class_list
        

    return (PA_with_param_class_list , PA_without_param_class_list)




def DTW_class_accuracy_old():
    map_data= defaultdict(list)
    for i in range(0,len(tab_dtw)-1):
        row1_idx1 = tab_dtw.iloc[i]['index1']
        row1_idx2 = tab_dtw.iloc[i]['index2']
        row1_dtw_val = tab_dtw.iloc[i]['dtw_value']

        for j in range(i,len(seg_df)):
            row2_idx1 = tab_dtw.iloc[j]['index1']
            row2_idx2 = tab_dtw.iloc[j]['index2']
            row2_dtw_val = tab_dtw.iloc[j]['dtw_value']
            if(row1_dtw_val == row2_dtw_val or row2_dtw_val - 0.03  <= row1_dtw_val <= row2_dtw_val + 0.03 ):
                map_data[i].append(row1_idx1)
                map_data[i].append(row1_idx2)
                map_data[i].append(row2_idx1)
                map_data[i].append(row2_idx2)

    map_data_temp1 = defaultdict(list)
    for k,val in map_data.items():
        unique_val = list(set(val))
        map_data_temp1[k].append(unique_val)
        
    PA_with_param_class_list=[]
    for k,val in map_data_temp1.items():
        PA_with_param_class_list.append(val[0])
        prep_dtw_vis(k,val[0],seg_df)
    return (PA_with_param_class_list )

#lisst = DTW_class_accuracy()


def DTW_class_accuracy():
    dtw_idx_full_list = tab_dtw['indices'].tolist()
    dtw_idx_list = dtw_idx_full_list[0:len(data_df)]
    
    dtw_indices=[]
    for i in dtw_idx_list:
        dtw_indices.extend(i)
    
    
    dtw_indices = list(set(dtw_indices))
    dtw_indices.sort()


    data_df_lst = data_df['index']
    data_df_lst =[i * 100 for i in data_df_lst]
    

    dtw_true_pos = 0
    dtw_false_pos = 0
    dtw_true_neg = 0
    for i in data_df_lst:
        if(i in dtw_indices):
            dtw_true_pos = dtw_true_pos + 1
        else:
            dtw_false_pos = dtw_false_pos + 1
    
    dtw_true_neg = len(dtw_idx_full_list) - len(dtw_idx_list)
    

    #print("DTW vis")
    #dtw_visualization_DTW(dtw_indices,seg_df)
    return (dtw_true_pos,dtw_false_pos, dtw_true_neg)







def eval_methods():
    df_org_seg = pd.DataFrame()
    df_org_class_temp = defaultdict(list)
    df_org_class = []
    map_data= defaultdict(list)
    
    
    df_topk = data_df.groupby(['class']).count()
    top_k = 0
    for i in range(len(df_topk)):
         top_k_temp1 =  df_topk.iloc[i]['data']
         top_k_temp2 = int(top_k_temp1*(top_k_temp1-1)/2)
         if(top_k_temp2 == 0):
             top_k = top_k + 1
         else:
             top_k = top_k + top_k_temp2
    

    for i in range(0,len(data_df)-1):
        
        row1_idx = data_df.iloc[i]['index']
        row1_class = data_df.iloc[i]['class']
        
        for j in range(i,len(data_df)):
            
            row2_idx = data_df.iloc[j]['index']
            row2_class = data_df.iloc[j]['class']
            
            indices=[]

            map_data[row1_class].append(row1_idx* window_size)
            map_data[row2_class].append(row2_idx* window_size)


            

            if( row1_class == row2_class and row1_idx != row2_idx):
                indices=[row1_idx* window_size,row2_idx* window_size]
                
                temp_df = pd.DataFrame([[row1_class,indices,row1_idx* window_size,row2_idx* window_size]],
                                       columns=['class','indices','index1','index2'])
                df_org_seg = df_org_seg.append(temp_df,ignore_index=True)

            df_org_class_temp[int(row1_class)].append(row1_idx* window_size)
            df_org_class_temp[int(row2_class)].append(row2_idx* window_size)
            

    
    
    
    
    map_data_list = []
    for k, v in map_data.items():
        unique_val = list(set(v))
        unique_val.sort()

        if(len(unique_val) > 1):
            for i in range(0,len(unique_val)-1):
                for j in range(i,len(unique_val)):
                    if(unique_val[i] != unique_val[j]):
                        indices=[unique_val[i],unique_val[j]]

                        map_data_list.append(indices)
        else:

            map_data_list.append(unique_val)
    
    
    
    
    
    
    
    """ ------ ------------------Segments -------------------"""
    
    proposed_idx_list = tab_PA_class_rank_without['indices'].tolist()
    data_idx =  df_org_seg['indices']


    dtw_idx_list = tab_dtw['indices'].tolist()
    PA_segment_accuracy = 0
    DTW_segment_accuracy = 0
    for idx in data_idx:
        if (idx in proposed_idx_list):
            PA_segment_accuracy = PA_segment_accuracy + 1
        
        if (idx in dtw_idx_list):
           DTW_segment_accuracy = DTW_segment_accuracy + 1
    
    
    
    """ ------ ------------------Class Accuracy  -------------------"""
    df_org_class_temp1 = defaultdict(list)
    for k,val in df_org_class_temp.items():
        unique_val = list(set(val))
        df_org_class_temp1[k].append(unique_val)
        

    for k,val in df_org_class_temp1.items():
        df_org_class.append(val[0])
    
    
    
    PA_with_param_class_list ,PA_without_param_class_list = PA_class_accuracy()
    
    dtw_true_pos,dtw_false_pos, dtw_true_neg = DTW_class_accuracy()
    PA_with_class_true_pos = 0
    PA_with_class_true_neg = 0
    PA_with_class_false_pos = 0
    PA_without_class_true_pos = 0
    PA_without_class_true_neg = 0
    PA_without_class_false_pos = 0
    
    for lst in df_org_class:
        
        if(lst in PA_with_param_class_list):
            PA_without_class_true_pos = PA_without_class_true_pos + 1
        else:
            PA_without_class_false_pos = PA_without_class_false_pos + 1
        
        if(lst in PA_without_param_class_list):
            PA_with_class_true_pos = PA_with_class_true_pos + 1
        else:
            PA_with_class_false_pos = PA_with_class_false_pos + 1
            

    
    
    PA_with_class_true_neg = len(PA_with_param_class_list) - PA_with_class_true_pos - PA_with_class_false_pos
    
    PA_without_class_true_neg = len(PA_without_param_class_list) - PA_without_class_true_pos - PA_without_class_false_pos
    
    
    
    PA_without_idx = tab_PA_class_rank_without['indices'].tolist()
    PA_without_idx = PA_without_idx[0:top_k]
    dtw_idx_list = dtw_idx_list[0:top_k]
    df_org_class_temp11 = df_org_class_temp
    PA_without_class_new_true_pos = 0
    PA_without_class_new_false_pos= 0
    PA_without_class_new_true_neg = 0
    
    
    
    per = math.floor(0.04 * top_k)
    
    DTW_class_new_true_pos = 0
    DTW_class_new_false_pos= 0
    DTW_class_new_true_neg = 0
    for key1, val1 in df_org_class_temp11.items():
        unique_val1 = list(set(val1))
        unique_val1.sort()
        data_vals = unique_val1
        
        if(len(data_vals) > 1):
            for i in range(0,len(data_vals)-1):
                for j in range(i,len(data_vals)):
                    if(data_vals[i] != data_vals[j]):
                        ind = [data_vals[i],data_vals[j]]
                        
                        
                        if(ind in PA_without_idx):
                            PA_without_class_new_true_pos = PA_without_class_new_true_pos + 1
                        else:
                            PA_without_class_new_false_pos = PA_without_class_new_false_pos + 1
                            
                        if(ind in dtw_idx_list):
                            DTW_class_new_true_pos = DTW_class_new_true_pos + 1
                        else:
                            DTW_class_new_false_pos = DTW_class_new_false_pos + 1
    PA_without_class_new_true_neg = per#len(tab_PA_class_rank_without) - PA_without_class_new_true_pos - PA_without_class_new_false_pos
    DTW_class_new_true_neg = len(tab_dtw) - DTW_class_new_true_pos - DTW_class_new_false_pos



    
    """-----------------------------Sub Class Accuracy -----------------------------"""
    scale_subClass_accuracy_with=[]
    for key, val in temp_subClass_scale_with.items():
        unique_val = list(set(val))
        scale_subClass_accuracy_with.append(unique_val)
        
        
    scale_subClass_accuracy_without=[]
    for key, val in temp_subClass_scale_without.items():
        unique_val = list(set(val))
        scale_subClass_accuracy_without.append(unique_val)
        
    org_subClass_list=[]
    for key1, val1 in map_data.items():
        unique_val1 = list(set(val1))
        org_subClass_list.append(unique_val1)
    
    PA_with_subClass_scale_true_pos = 0
    PA_with_subClass_scale_false_pos = 0
    PA_with_subClass_scale_true_neg = 0
    PA_without_subClass_scale_true_pos = 0
    PA_without_subClass_scale_false_pos = 0
    PA_without_subClass_scale_true_neg = 0
    for lst in org_subClass_list:
        if(lst in scale_subClass_accuracy_without):
            PA_without_subClass_scale_true_pos = PA_without_subClass_scale_true_pos + 1
        else:
            PA_without_subClass_scale_false_pos = PA_without_subClass_scale_false_pos + 1
        
        if(lst in scale_subClass_accuracy_with):
            PA_with_subClass_scale_true_pos = PA_with_subClass_scale_true_pos + 1
        else:
            PA_with_subClass_scale_false_pos = PA_with_subClass_scale_false_pos + 1
    
    PA_with_subClass_scale_true_neg = len(scale_subClass_accuracy_with) - PA_with_subClass_scale_true_pos - PA_with_subClass_scale_false_pos
    PA_without_subClass_scale_true_neg = len(scale_subClass_accuracy_without) - PA_without_subClass_scale_true_pos - PA_without_subClass_scale_false_pos
    

    
    PA_with_idx = sub_class_idx()
    PA_with_idx = PA_with_idx[0:top_k]

    PA_with_subClass_new_true_pos = 0
    PA_with_subClass_new_false_pos= 0
    PA_with_subClass_new_true_neg = 0

    
    DTW_subClass_new_true_pos = 0
    DTW_subClass_new_false_pos= 0
    DTW_subClass_new_true_neg = 0
    for ind in map_data_list:
        
        if(ind in PA_with_idx):
            PA_with_subClass_new_true_pos = PA_with_subClass_new_true_pos + 1
        else:
            PA_with_subClass_new_false_pos = PA_with_subClass_new_true_pos + 1
                            
        if(ind in dtw_idx_list):
            DTW_subClass_new_true_pos = DTW_subClass_new_true_pos + 1
        else:
            DTW_subClass_new_false_pos = DTW_subClass_new_false_pos + 1
                            
    PA_with_subClass_new_true_neg = per#len(tab_PA_class_rank_without) - PA_with_subClass_new_true_pos - PA_with_subClass_new_true_pos
    DTW_subClass_new_true_neg = len(tab_dtw) - DTW_subClass_new_false_pos - DTW_subClass_new_false_pos
    
    
    
    
    """---------------------------------- Report ------------------------------------"""
    print("")
    print("")
    print("")
    print("")
    print("")
    print("Dataset Size : ", len(ts))
    print("")
    print("---------------Segment Accuracy---------------------")
    print("Actual Segments                      :", len(data_idx))
    print("PA Segment Accuracy                  :", PA_segment_accuracy)
    print("DTW  Segment Accuracy                :", DTW_segment_accuracy)
    print("")
    print("---------------Class Accuracy---------------------")
    print("Actual class true pos                :", len(df_org_class))
    print("------- DTW ----")
    print("DTW class true pos                   :", dtw_true_pos)
    print("DTW class false pos                  :", dtw_false_pos)
    print("DTW class  true neg                  :", dtw_true_neg)
    print("------- PA without ----")
    print("PA without class true pos            :", PA_without_class_true_pos)
    print("PA without class false pos           :", PA_without_class_false_pos)
    print("PA without class true neg            :", PA_without_class_true_neg)
    print("------- PA with ----")
    print("PA with class true pos               :", PA_with_class_true_pos)
    print("PA with class false pos              :", PA_with_class_false_pos)
    print("PA with class true neg               :", PA_with_class_true_neg)
    print("---------------Sub Class Accuracy---------------------")
    print("Actual Sub Classes                   :", len(org_subClass_list))
    print("------- PA without ----")
    print("PA without Scale SubClass true pos   :", PA_without_subClass_scale_true_pos)
    print("PA without Scale SubClass false pos  :", PA_without_subClass_scale_false_pos)
    print("PA without Scale SubClass true neg   :", PA_without_subClass_scale_true_neg)
    print("------- PA with ----")
    print("PA with Scale SubClass true pos      :", PA_with_subClass_scale_true_pos)
    print("PA with Scale SubClass false pos     :", PA_with_subClass_scale_false_pos)
    print("PA with Scale SubClass true neg      :", PA_with_subClass_scale_true_neg)

    print("")
    print("-------------------- TOP K-------------------------")
    print("Top k:",top_k)
    print("------- PA without TOPK ----")
    print("PA without class top K true pos      :", PA_without_class_new_true_pos)
    print("PA without class top K false pos     :", PA_without_class_new_false_pos)
    print("PA without class top K true neg      :", PA_without_class_new_true_neg)
    print("------- DTW Class TOPK ----")
    print("DTW top K class true pos             :", DTW_class_new_true_pos)
    print("DTW top K class false pos            :", DTW_class_new_false_pos)
    print("DTW top K class  true neg            :", DTW_class_new_true_neg)
    print("------- PA with TOPK ----")
    print("PA with Sub Class top K true pos     :", PA_with_subClass_new_true_pos)
    print("PA with Sub Class top K false pos    :", PA_with_subClass_new_false_pos)
    print("PA with Sub Class top K true neg     :", PA_with_subClass_new_true_neg)
    print("------- DTW Sub Class TOPK ----")
    print("DTW top K Sub Class true pos          :", DTW_subClass_new_true_pos)
    print("DTW top K Sub Class false pos         :", DTW_subClass_new_false_pos)
    print("DTW top K Sub Class  true neg         :", DTW_subClass_new_true_neg)
    
    
    
    

    return (map_data,df_org_class,org_subClass_list,scale_subClass_accuracy_without,scale_subClass_accuracy_with )




eval_start = timeit.default_timer()
#tab_segments_org, tab_class_org, tab_subClass_org,tab_subClass_PA_without,tab_subClass_PA_with = eval_methods()
eval_stop = timeit.default_timer()



