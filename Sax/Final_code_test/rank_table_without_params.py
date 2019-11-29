import pandas as pd
from collections import defaultdict
from helper_functions import dtw_val_gen , dtw_rank_gen
from dtw_visualization import dtw_visualization1 ,dtw_visualization2,dtw_visualization3



def  rank_table_without_param (df_dtw_prep, compare_list, window_size, ts):

    dtw_rank_df=pd.DataFrame()
    map_scale_temp = defaultdict(list)
    total_k = len(compare_list)
    for k, v in compare_list.items():
        
        print(total_k,"------", k)
        v_temp=str(v)[2:-2]
        v1=[int(s) for s in v_temp.split(',')]
        
        if(len(v1) > 1):
            dtw_temp=pd.DataFrame()
            print("Length of Comapre List",len(v1))
            for i in range(0,len(v1)-1):
                print(i+1)
                for j in range(i,len(v1)):
                    if(v1[i] != v1[j]):
                        row1 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[i]]
                        row2 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[j]]
                        
                        sub_section1 = row1.iloc[0]['sub_section']
                        sub_section2 = row2.iloc[0]['sub_section']

                        index1 = row1.iloc[0]['indices']
                        index2 = row2.iloc[0]['indices']
                        indices =[]
                        indices=[index1,index2]
                        
                        
                        map_scale_temp[k].append(index1)
                        map_scale_temp[k].append(index2)
                        
                        dtw_value= dtw_val_gen(sub_section1, sub_section2,0)
                        temp_df = pd.DataFrame([[k,index1,index2,indices,dtw_value,sub_section1,sub_section2]], 
                                               columns=['key','index1','index2','indices','dtw_value','sub_section1','sub_section2'])
                        
                        dtw_temp=dtw_temp.append(temp_df,ignore_index=True)

            dtw_temp = dtw_rank_gen(dtw_temp)
            dtw_rank_df= dtw_rank_df.append(dtw_temp,ignore_index=True)

            #prep_visualize1(dtw_temp,window_size,ts,df_dtw_prep)
             
        else:
            dtw_temp=pd.DataFrame()
            #print(k)
            for i in range(0,len(v1)):

                
                row1 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[i]]
                sub_section1 = row1.iloc[0]['sub_section']
                index1 = row1.iloc[0]['indices']
                indices = [index1]
                
                
                temp_df = pd.DataFrame([[k,index1,indices,sub_section1]], columns=['key','index1','indices','sub_section1'])
                dtw_temp=dtw_temp.append(temp_df,ignore_index=True)
                
                map_scale_temp[k].append(index1)
            
            dtw_rank_df= dtw_rank_df.append(dtw_temp,ignore_index=True)
            #prep_visualize2(dtw_temp)
        
        total_k = total_k - 1


    tab_proposed = dtw_rank_df.sort_values(by=['dtw_value'])
    return(tab_proposed ,map_scale_temp)
    


def prep_visualize1(dtw_temp,window_size,ts,df_dtw_prep):
        #dtw_visualization(dtw_temp,window_size, ts)
        dtw_visualization1(dtw_temp,df_dtw_prep)
        
def prep_visualize2(dtw_temp):
        dtw_visualization2(dtw_temp)
        