import numpy as np
import pandas as pd


from matrix_calc import matrix_prep




"""-------------     Intialization     ------------- """
y_alphabet_size=4
word_lenth=3
window_size=10
skip_offset=2
ham_distance=1
epsilon = 1e-6

data =  pd.read_csv('ecg.csv', sep=',', header=None)
x1 = data.iloc[1:,1].values.flatten() 
x1=np.asfarray(x1,float)


matrix_prep(x1,window_size,skip_offset,word_lenth,ham_distance)