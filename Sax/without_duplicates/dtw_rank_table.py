import pandas as pd
from helper_functions import dtw_val_gen , dtw_rank_gen
from dtw_visualization import dtw_visualization ,dtw_visualization2





def  dtw_rank_table (df_dtw_prep, compare_list, window_size, ts):

    dtw_rank_df=pd.DataFrame()
    test_arr=[]
    
    for k, v in compare_list.items():

        v_temp=str(v)[2:-2]
        v1=[int(s) for s in v_temp.split(',')]
        
        if(len(v1) > 1):
            dtw_temp=pd.DataFrame()
            print(k)
            for i in range(0,len(v1)-1):
                for j in range(i,len(v1)):
                    if(v1[i] != v1[j]):
                        row1 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[i]]
                        row2 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[j]]
                        
                        sub_section1 = row1.iloc[0]['sub_section']
                        sub_section2 = row2.iloc[0]['sub_section']
                        
                        index1 = row1.iloc[0]['indices']
                        index2 = row2.iloc[0]['indices']
                        
                        subs=[]
                        subs.append(index1)
                        subs.append(index2)
                        
                        if(subs  not in test_arr):
                            dtw_value= dtw_val_gen(sub_section1, sub_section2,0)
                            temp_df = pd.DataFrame([[k,index1,index2,sub_section1,sub_section2,dtw_value]], columns=['key','index1','index2','sub_section1','sub_section2','dtw_value'])
                            dtw_temp=dtw_temp.append(temp_df,ignore_index=True)
                            
                        
                        test_arr.append(subs)

            #dtw_temp = dtw_rank_gen(dtw_temp)
            dtw_rank_df= dtw_rank_df.append(dtw_temp,ignore_index=True)
            
            
             
        else:
            dtw_temp=pd.DataFrame()
            print(k)
            for i in range(0,len(v1)-1):
                for j in range(i,len(v1)):

                    if(v1[i] != v1[j]):
                        row1 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[i]]
                        row2 = df_dtw_prep.loc[df_dtw_prep['indices'] == v1[j]]
                        
                        sub_section1 = row1.iloc[0]['sub_section']
                        sub_section2 = row2.iloc[0]['sub_section']

                        index1 = row1.iloc[0]['indices']
                        index2 = row2.iloc[0]['indices']
                        
                        
                        temp_df = pd.DataFrame([[k,index1,index2,sub_section1,sub_section2]], columns=['key','index1','index2','sub_section1','sub_section2'])
                        dtw_temp=dtw_temp.append(temp_df,ignore_index=True)

            
            dtw_rank_df= dtw_rank_df.append(dtw_temp,ignore_index=True)
            
        
        
        #prep_visualize(dtw_temp,window_size,ts,df_dtw_prep)
        


    return(dtw_rank_df)
    


def prep_visualize(dtw_temp,window_size,ts,df_dtw_prep):
    if(len(dtw_temp)> 0):
        #dtw_visualization(dtw_temp,window_size, ts)
        dtw_visualization2(dtw_temp,df_dtw_prep)
        