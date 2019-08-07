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

plt.plot(range(20))
plt.axvspan(0, 20, color='red', alpha=0.3)

plt.axvspan(2, 4, color='red', alpha=0.5)
plt.axvspan(5, 9, color='red', alpha=0.5)
plt.axvspan(5, 6, color='red', alpha=0.5)
plt.show()





