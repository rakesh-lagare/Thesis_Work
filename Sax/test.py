# -*- coding: utf-8 -*-
"""
Created on Oct 5 19:26:55 2018

@author: Meagatron
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pylab as plt


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

"""
import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(1, 2, 1)
plt.plot(x1, y1, 'ko-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')




plt.subplot(2, 2, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()




for i in range(1, 7):
    plt.subplot(2, 3, i)
    #plt.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')
    plt.plot(x2, y2, 'r.-')
    plt.xlabel('time (s)')
    plt.ylabel('Undamped')

"""

#First create some toy data:
x = np.random.randn(20)
y = np.random.randn(20)

fig = plt.figure(figsize=(15, 20))

for i in range(0,10):
 #if n % 2 == 0:
    ax = fig.add_subplot(5, 3,i+1 )
    plt.plot(x,y)
 #else:
    #ax = fig.add_subplot(int(n/2)+1, 2)


plt.show()

