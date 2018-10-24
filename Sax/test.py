# -*- coding: utf-8 -*-
"""
Created on Oct 5 19:26:55 2018

@author: Meagatron
"""

import matplotlib.pyplot as plt
import numpy as np

"""



plt.plot(x, 'ro')
plt.plot(x[3:5],  'g*')

plt.show()

x = [1,2,3,4,5,6]
y = [3,4,1,2,3,4]
z= [4,5,6]

plt.plot(x)
plt.plot(y[2:5],lw=10, c='yellow', zorder=-1)
plt.plot(z,lw=10, c='red', zorder=-1)

"""

x = np.linspace(-10, 10, 100)
#plt.plot(x)
#aa=x[60:80]
#plt.plot(x[60:80], lw=10, c='yellow', zorder=-1)


nnn=min([1,2,3,4,5,5.1], key=lambda x:abs(x-5.09))

import numpy as np

def split_into_parts(number, n_parts):
    return np.linspace(0, number, n_parts+1)[1:]

aa=split_into_parts(1, 3)


median=sorted(aa)[len(aa) // 2]


documents = [['Human machine interface for lab abc computer applications','4'],
             ['A survey of user opinion of computer system response time','3'],
             ['The EPS user interface management system','2']]


documents = [sub_list[0] for sub_list in documents]


s = ["this", "this", "and", "that"]

        print (s[i])








