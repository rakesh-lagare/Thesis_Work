from random import randrange
import matplotlib.pyplot as plt
import numpy as np


tsn= [10,10,10,10,13,10,10,10,13,10,10,10]

ts0= [10,20,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,20,10]

ts1 = [10,10,20,40,60,75,80,75,10,10,20,40,60,75,80,75,10,10,20,40,60,75,80,75,10,10]

ts2 = [10,10,20,30,40,50,60,70,80,90,100,110,120,110,100,90,80,70,60,50,40,30,20,10,10]


def gen_ts_data(num):
    ts_data = []
    for _ in range(num):
        if(randrange(3)==0):
            ts_data.extend(tsn+ts0+ tsn)
        elif (randrange(3)==1):
            ts_data.extend(tsn+ ts1 + tsn)
        else:
            ts_data.extend(tsn+ ts2 + tsn)
    return ts_data



ts = np.asfarray(gen_ts_data(5),float)


plt.plot(ts)
plt.show()





#tsc_temp= tsn+ts2+ tsn
#tsc = np.asfarray(tsc_temp,float)