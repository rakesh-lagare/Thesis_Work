import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtw import dtw
from statistics import median


def dtw_val_gen(sub_section1,sub_section2,dt):
    if (dt == 0): #Normal DTW
        x=np.array(sub_section1).reshape(-1, 1)
        y=np.array(sub_section2).reshape(-1, 1)
        euclidean_norm = lambda x, y: np.abs(x - y)
        dtw_value, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
    else:   #Fast DTW
        x = np.array(sub_section1)
        y = np.array(sub_section2)
        dtw_value, path = fastdtw(x, y, dist=euclidean)

    return dtw_value




def dtw_rank_gen(dtw_temp):
    
    med=(dtw_temp['dtw_value'] ).tolist()
    if(len(dtw_temp)> 5) :
        dtw_temp = dtw_temp[dtw_temp['dtw_value'] < median(med)  ]  #median(med)
    dtw_temp= dtw_temp.sort_values(by=['dtw_value'])
    rank_list=[]
    for m in range(1, len(dtw_temp)+1):
            rank_list.append(m)
    dtw_temp.insert(loc=6, column='ranks', value=rank_list)
    
    return dtw_temp



"""-------------     Y_Alphabetize      ------------- """
def alphabetize_ts(sub_section,y_alpha_size):
    y_alphabets=get_y_alphabets(y_alpha_size)
    mean_val=x_distrubted_values(sub_section)
    y_alpha_val=min(y_alphabets, key=lambda x:abs(x-mean_val))
    y_alpha_idx=y_alphabets.index(y_alpha_val)
    curr_word = index_to_letter(y_alpha_idx)

    return(curr_word)

"""-------------   index to letter      ------------- """


def index_to_letter(idx):
    """Convert a numerical index to a char."""
    if 0 <= idx < 20:
        return chr(97 + idx)
    else:
        raise ValueError('A wrong idx value supplied.')


"""-------------     X-axis Distribution      ------------- """

def x_distrubted_values(series):
    mean=np.mean(series)
    median=sorted(series)[len(series) // 2]
    return median


"""-------------     Normalization      ------------- """

def normalize(x):
        epsilon = 1e-6
        X = np.asanyarray(x)
        if np.nanstd(X) < epsilon:
            res = []
            for entry in X:
                if not np.isnan(entry):
                    res.append(0)
                else:
                    res.append(np.nan)
            return res
        return (X - np.nanmean(X)) / np.nanstd(X)

def normal_distribution(x):
    x = (x-min(x))/(max(x)-min(x))
    return x

"""-------------     Y-axis Distribution      ------------- """
def break_points_gaussian(size):
    options = {
            3: np.array([ -0.43, 0.43]),
            4: np.array([ -0.67, 0, 0.67]),
            5: np.array([ -0.84, -0.25, 0.25, 0.84]),
            6: np.array([ -0.97, -0.43, 0, 0.43, 0.97]),
            7: np.array([ -1.07, -0.57, -0.18, 0.18, 0.57, 1.07]),
            8: np.array([ -1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15]),
            9: np.array([ -1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22]),
            10: np.array([ -1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]),
            11: np.array([ -1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34]),
            12: np.array([ -1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38]),
            13: np.array([ -1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43]),
            14: np.array([ -1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47]),
            15: np.array([ -1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5]),
            16: np.array([ -1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53]),
            17: np.array([ -1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56]),
            18: np.array([ -1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59]),
            19: np.array([ -1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62]),
            20: np.array([ -1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]),
            }

    return options[size]


"""-------------     Get y_alphabets      ------------- """

def get_y_alphabets(y_alpha_size):
    y_alpha_size
    #y_alphabets = break_points_quantiles(y_alphabet_size).tolist()
    y_alphabets = break_points_gaussian(y_alpha_size).tolist()
    return y_alphabets



"""-------------     Hamming Distance      ------------- """
def hamming_distance1(string1, string2):
    distance = 0
    L = len(string1)
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))
