from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import math   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



pre_data =  pd.read_csv('ECG200.csv', sep=',', header=None)
pre_data = pre_data[:500]
ts = pre_data.iloc[:,0].values.flatten()

plt.plot(ts)
plt.show()