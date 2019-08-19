import matplotlib.pyplot as plt


def dtw_visualization(dtw_df,skip_offset,ts):

    idx1 = dtw_df['index1'].tolist()
    idx2 = dtw_df['index2'].tolist()
    
    idx= idx1 + idx2
    unique_list = list(set(idx))
    print(unique_list)
    
    
    plt.figure(figsize=(16,10), dpi= 60)
    plt.plot(ts)

    for i in unique_list:
        start_idx = i
        end_idx= i + skip_offset
        plt.axvspan(start_idx, end_idx, color='red', alpha=0.4)
        
    plt.show()
        




def dtw_visualization2(dtw_df,seg_df):
   
    idx1 = dtw_df['index1'].tolist()
    idx2 = dtw_df['index2'].tolist()
    idx= idx1 + idx2
    unique_list = list(set(idx))
    lent= len(unique_list)
    row=int(lent/4)
    
    print(unique_list)
    
    if(lent > 4):
        fig = plt.figure(figsize=(6*row, 5*row))
        for i in range(0,lent):
            row1 = seg_df.loc[seg_df['indices'] == unique_list[i]]
            sub_section = row1.iloc[0]['sub_section']
            fig.add_subplot(row+1, 4,i+1 )
            plt.plot(sub_section, '--.')
    else:
        fig = plt.figure(figsize=(3*3, 4*3))
        for i in range(0,lent):
            row1 = seg_df.loc[seg_df['indices'] == unique_list[i]]
            sub_section = row1.iloc[0]['sub_section']
            fig.add_subplot(5, 2,i+1 )
            plt.plot(sub_section)
    #plt.savefig('./Output/sliding_half_segment/'+key+'.png')
    #plt.savefig('books_read.png')        
    plt.show()