# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:58:38 2019

@author: Meagatron
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from random import randrange

from fastdtw import fastdtw

import matplotlib.pyplot as plt



ts1= [2,4,8,6,4]
ts2= [2,3,2,2,3,3,2,2,3,2,3]
ts= ts1+ ts2
ts = np.asfarray(ts,float)


print(abs(1-2))
#plt.plot(ts)
#plt.show()

