
import matplotlib.pyplot as plt
import os


def dtw_visualization1(dtw_df,seg_df):
    if(len(dtw_df)> 0):
        idx1 = dtw_df['index1'].tolist()
        idx2 = dtw_df['index2'].tolist()
        idx= idx1 + idx2
        unique_list = list(set(idx))
        lent= len(unique_list)
        row=int(lent/4)
        key = dtw_df.iloc[0]['key']
        #print(key)
        #print(unique_list)
        
        
        if(lent > 4):
            fig = plt.figure(figsize=(3*3, 4*3))
            for i in range(0,lent):
                row1 = seg_df.loc[seg_df['indices'] == unique_list[i]]
                sub_section = row1.iloc[0]['sub_section']
                #key = row1.iloc[0]['key']
                fig.add_subplot(row+1, 4,i+1 )
                plt.plot(sub_section)
                #plt.plot(sub_section, '--.')
                
        else:
            fig = plt.figure(figsize=(3*3, 4*3))
            for i in range(0,lent):
                row1 = seg_df.loc[seg_df['indices'] == unique_list[i]]
                sub_section = row1.iloc[0]['sub_section']
                fig.add_subplot(5, 2,i+1 )
                #plt.plot(sub_section, '--.')
                plt.plot(sub_section)

        plt.savefig('./Output/without_param/' +key+'.png')        
        plt.show()
        

def dtw_visualization2(dtw_df):
        key = dtw_df.iloc[0]['key']
        #print(key)
        #print(dtw_df.iloc[0]['index1'])
        sub_section = dtw_df.iloc[0]['sub_section1']
        #plt.plot(sub_section, '--.')
        plt.plot(sub_section)
        plt.savefig('./Output/without_param/' +key+'.png')        
        plt.show()
        

        
def dtw_visualization3(dtw_df,skip_offset,ts):

    idx1 = dtw_df['index1'].tolist()
    idx2 = dtw_df['index2'].tolist()
    
    idx= idx1 + idx2
    unique_list = list(set(idx))
    #print(unique_list)
    
    
    plt.figure(figsize=(16,10), dpi= 60)
    plt.plot(ts)

    for i in unique_list:
        start_idx = i
        end_idx= i + skip_offset
        plt.axvspan(start_idx, end_idx, color='red', alpha=0.4)
        
    plt.show()
        

def dtw_visualization_scale(key,idx,seg_df):


        unique_list = list(set(idx))
        lent= len(unique_list)
        row=int(lent/4)
        
        #print(key)
        #print(unique_list)
        
        
        if(lent > 4):
            fig = plt.figure(figsize=(3*3, 4*3))
            for i in range(0,lent):
                row1 = seg_df.loc[seg_df['indices'] == unique_list[i]]
                sub_section = row1.iloc[0]['sub_section']
                fig.add_subplot(row+1, 4,i+1 )
                plt.plot(sub_section)
                #plt.plot(sub_section, '--.')
                
        else:
            fig = plt.figure(figsize=(3*3, 4*3))
            for i in range(0,lent):
                row1 = seg_df.loc[seg_df['indices'] == unique_list[i]]
                sub_section = row1.iloc[0]['sub_section']
                fig.add_subplot(5, 2,i+1 )
                #plt.plot(sub_section, '--.')
                plt.plot(sub_section)

        plt.savefig('./Output/with_param/' +key+'.png')        
        plt.show()


def dtw_visualization_scale2(dtw_df):
        key = dtw_df.iloc[0]['key']
        
        #print(key)
        #print(dtw_df.iloc[0]['index1'])
        
        sub_section = dtw_df.iloc[0]['sub_section1']
        #plt.plot(sub_section, '--.')
        plt.plot(sub_section)
        plt.savefig('./Output/with_param/' +key+'.png')        
        plt.show()









def dtw_visualization_DTW(idx,seg_df):


        lent= len(idx)
        row=int(lent/4)
        
        #print(key)
        #print(unique_list)
        
        
        if(lent > 4):
            fig = plt.figure(figsize=(3*3, 4*3))
            for i in range(0,lent):
                row1 = seg_df.loc[seg_df['indices'] == idx[i]]
                sub_section = row1.iloc[0]['sub_section']
                fig.add_subplot(row+1, 4,i+1 )
                plt.plot(sub_section)
                plt.savefig('./Output/DTW/' +str(i)+'.png')
                plt.show()
                #plt.plot(sub_section, '--.')
                
        else:
            fig = plt.figure(figsize=(3*3, 4*3))
            for i in range(0,lent):
                row1 = seg_df.loc[seg_df['indices'] == idx[i]]
                sub_section = row1.iloc[0]['sub_section']
                fig.add_subplot(5, 2,i+1 )
                #plt.plot(sub_section, '--.')
                plt.plot(sub_section)
                plt.savefig('./Output/DTW/' +str(i)+'.png')
                plt.show()

        
        






def prep_dtw_vis(key,idx,seg_df):
        key = str(key)
        unique_list = list(set(idx))
        lent= len(unique_list)
        row=int(lent/4)
        
        #print(key)
        #print(unique_list)
        
        
        if(lent > 4):
            fig = plt.figure(figsize=(3*3, 4*3))
            for i in range(0,lent):
                row1 = seg_df.loc[seg_df['indices'] == unique_list[i]]
                sub_section = row1.iloc[0]['sub_section']
                fig.add_subplot(row+1, 4,i+1 )
                plt.plot(sub_section)
                #plt.plot(sub_section, '--.')
                
        else:
            fig = plt.figure(figsize=(3*3, 4*3))
            for i in range(0,lent):
                row1 = seg_df.loc[seg_df['indices'] == unique_list[i]]
                sub_section = row1.iloc[0]['sub_section']
                fig.add_subplot(5, 2,i+1 )
                #plt.plot(sub_section, '--.')
                plt.plot(sub_section)

        plt.savefig('./Output/DTW/' +key+'.png')        
        plt.show()