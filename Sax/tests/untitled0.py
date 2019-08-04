# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:58:38 2019

@author: Meagatron
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw



x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)
