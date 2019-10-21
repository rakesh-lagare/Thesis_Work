import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict
from segmentation import segment_ts,dtw_segment_ts
from compare_shape import compare_shape_algo
from rank_table_with_params import rank_table_with_param
from rank_table_without_params import rank_table_without_param
from helper_functions import normalize
from evaluation import orinal_dtw_rank_tab,euclidean_rank_tab
from dtw_visualization import prep_dtw_vis





"""-------------     Intialization     ------------- """
start = timeit.default_timer()


""" ------------ OLD
pre_data =  pd.read_csv('dataList.csv', sep=',', header=None)
#pre_data = pre_data[:1000]
#ts = pre_data.iloc[1:,1].values.flatten()

#ts= gen_ts_data(10)

#data_df,ts = prep_data(10)

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

remove_files()

pre_data =  pd.read_csv('dataList.csv', sep=',', header=None)
data_df = pd.read_csv('dataframe.csv', sep=',' )
ts = pre_data.iloc[1:, :1]

ts = np.asfarray(ts,float)
ts= normalize(ts)

plt.plot(ts)
plt.savefig('./Output/Original.png')
plt.show()

 
y_alpha_size = 4
word_length = 3
window_size = 100#round( len(ts) * 0.1 )
skip_offset = round(window_size   ) 
ham_distance = 0



seg_alpha, seg_indices,seg_df = segment_ts(ts,window_size,skip_offset,word_length,y_alpha_size)
seg_dtw_df = dtw_segment_ts(ts,window_size,skip_offset)
compare_strings, compare_list = compare_shape_algo(seg_alpha,seg_indices,seg_df,word_length,ham_distance)


#print("-------------------------------Proposed--------------------------")
print("--------------   Without Param  ----------------------------------")
tab_PA_class_rank_without,temp_subClass_scale_without = rank_table_without_param(seg_df, compare_list, window_size, ts)

print("--------------   With Param  ----------------------------------")
tab_PA_class_rank_with , temp_subClass_scale_with= rank_table_with_param(seg_df, compare_list, window_size, ts)


#print("--------------------------------Original DTW---------------------")
tab_dtw  = orinal_dtw_rank_tab(seg_dtw_df)



#print("----------------------------Euclidean---------------------------")
#tab_euclidean = euclidean_rank_tab(seg_df)



stop = timeit.default_timer()
print('Time: ', stop - start)  




""" ----------------------------  Evaluation -----------------------------------------"""


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




def DTW_class_accuracy():
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


def eval_methods():
    df_org_seg = pd.DataFrame()
    df_org_class_temp = defaultdict(list)
    df_org_class = []
    map_data= defaultdict(list)
    map_data_org_class= defaultdict(list)
    for i in range(0,len(data_df)-1):
        
        row1_idx = data_df.iloc[i]['index']
        row1_class = data_df.iloc[i]['class']

        for j in range(i,len(data_df)):
            
            row2_idx = data_df.iloc[j]['index']
            row2_class = data_df.iloc[j]['class']
            
            indices=[]
            #indices_new=[row1_idx* window_size,row2_idx* window_size]

            map_data[row1_class].append(row1_idx* window_size)
            map_data[row2_class].append(row2_idx* window_size)
            #if( row1_idx != row2_idx):
                #map_data_org_class[row1_class].append(indices_new)
                
                
            if( row1_class == row2_class and row1_idx != row2_idx):
                indices=[row1_idx* window_size,row2_idx* window_size]
                
                temp_df = pd.DataFrame([[row1_class,indices,row1_idx* window_size,row2_idx* window_size]], columns=['class','indices','index1','index2'])
                df_org_seg = df_org_seg.append(temp_df,ignore_index=True)
                
 
            df_org_class_temp[int(row1_class)].append(row1_idx* window_size)
            df_org_class_temp[int(row2_class)].append(row2_idx* window_size)
                
                
    
    
    
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
    dtw_list = DTW_class_accuracy()
    PA_class_accuracy_with = 0
    PA_class_accuracy_without = 0
    dtw_class_accuracy = 0
    for lst in df_org_class:
        
        if(lst in PA_with_param_class_list):
            PA_class_accuracy_without = PA_class_accuracy_without + 1
        
        if(lst in PA_without_param_class_list):
            PA_class_accuracy_with = PA_class_accuracy_with + 1
            
        if(lst in dtw_list):
            dtw_class_accuracy = dtw_class_accuracy + 1
    
    

    
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
    
    PA_scale_subClass_accuracy_with = 0
    PA_scale_subClass_accuracy_without = 0
    for lst in org_subClass_list:
        if(lst in scale_subClass_accuracy_without):
            PA_scale_subClass_accuracy_without = PA_scale_subClass_accuracy_without + 1
        
        if(lst in scale_subClass_accuracy_with):
            PA_scale_subClass_accuracy_with = PA_scale_subClass_accuracy_with + 1
    
    
    
    """---------------------------------- Report ------------------------------------"""
    print("")
    print("")
    print("")
    print("")
    print("")
    print("Dataset Size : ", len(ts))
    print("")
    print("---------------Segment Accuracy---------------------")
    print("Actual Segments                    :", len(data_idx))
    print("PA Segment Accuracy                :", PA_segment_accuracy)
    print("DTW  Segment Accuracy              :", DTW_segment_accuracy)
    print("")
    print("---------------Class Accuracy---------------------")
    print("Actual classes                     :", len(df_org_class))
    print("DTW class accuracy                 :", dtw_class_accuracy)
    print("PA class accuracy without          :", PA_class_accuracy_without)
    print("PA class accuracy with             :", PA_class_accuracy_with)
    print("")
    print("---------------Sub Class Accuracy---------------------")
    print("Actual Sub Classes                 :", len(org_subClass_list))
    print("PA Scale SubClass Accuracy without :", PA_scale_subClass_accuracy_without)
    print("PA Scale SubClass Accuracy with    :", PA_scale_subClass_accuracy_with)
    

    return (df_org_seg,df_org_class,org_subClass_list,scale_subClass_accuracy_without,scale_subClass_accuracy_with )

eval_start = timeit.default_timer()
tab_segments_org, tab_class_org, tab_subClass_org,tab_subClass_PA_without,tab_subClass_PA_with = eval_methods()
eval_stop = timeit.default_timer()
print("")
print('Eval Time: ', eval_stop - eval_start)  


