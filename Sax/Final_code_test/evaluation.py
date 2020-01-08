import pandas as pd
import numpy as np
from saxpy.distance import euclidean
from helper_functions import dtw_val_gen
from dtw_visualization import dtw_visualization1,dtw_visualization2

def orinal_dtw_rank_tab(seg_df):
    dtw_temp=pd.DataFrame()
    print("Length of Data",len(seg_df))
    for i in range(0,len(seg_df)-1):
        #print(i+1)
        for j in range(i,len(seg_df)):
            print(i+1,"--- ",j+1)
            row1 = seg_df.loc[i]
            row2 = seg_df.loc[j]
            
            
                        
            sub_section1 = row1['sub_section']
            sub_section2 = row2['sub_section']

            index1 = row1['indices']
            index2 = row2['indices']
            if(index1 != index2):
                indices =[]
                indices=[index1,index2]
                dtw_value= dtw_val_gen(sub_section1, sub_section2,1)
                temp_df = pd.DataFrame([[index1,index2,indices,dtw_value]], 
                                               columns=['index1','index2','indices','dtw_value'])
                        
                dtw_temp=dtw_temp.append(temp_df,ignore_index=True)
    tab_dtw = dtw_temp.sort_values(by=['dtw_value'])
    return tab_dtw


def euclidean_rank_tab(seg_df):
    dtw_temp=pd.DataFrame()
    for i in range(0,len(seg_df)-1):
        for j in range(i,len(seg_df)):
            row1 = seg_df.loc[i]
            row2 = seg_df.loc[j]
            
            
                        
            sub_section1 = row1['sub_section']
            sub_section2 = row2['sub_section']

            index1 = row1['indices']
            index2 = row2['indices']
            if(index1 != index2):
                eucl_dist= np.linalg.norm(sub_section1-sub_section2)
                temp_df = pd.DataFrame([[index1,index2,sub_section1,sub_section2,eucl_dist]], columns=['index1','index2','sub_section1','sub_section2','eucl_dist'])
                dtw_temp=dtw_temp.append(temp_df,ignore_index=True)

    return dtw_temp            