import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt


from segmentation import segment_ts
from compare_shape import compare_shape_algo
from dtw_rank_table import dtw_rank_table
from helper_functions import normalize
from evaluation import orinal_dtw_rank_tab,euclidean_rank_tab





"""-------------     Intialization     ------------- """
start = timeit.default_timer()

pre_data =  pd.read_csv('ecg.csv', sep=',', header=None)
#pre_data = pre_data[:1000]
ts = pre_data.iloc[1:,1].values.flatten() 
ts = np.asfarray(ts,float)
ts= normalize(ts)

plt.plot(ts)
plt.show()

 
y_alpha_size = 4
word_length = 3
window_size = round( len(ts) * 0.1 )
skip_offset = round(window_size * 0.5)
ham_distance = 0



seg_alpha, seg_indices,seg_df = segment_ts(ts,window_size,skip_offset,word_length,y_alpha_size)

compare_strings, compare_list = compare_shape_algo(seg_alpha,seg_indices,seg_df,word_length,ham_distance)

dtw_tab = dtw_rank_table(seg_df, compare_list, window_size, ts)

#org_dtw_tab  = orinal_dtw_rank_tab(seg_df)
#euclidean_tab = euclidean_rank_tab(seg_df)





stop = timeit.default_timer()
print('Time: ', stop - start)  




