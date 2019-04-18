import numpy as np
import pandas as pd
import math
import itertools
import os





"""-------------     import Data     -------------"""









def do_local():
        data =  pd.read_csv('files/data.csv', header=None,usecols=[1],skiprows=1)
        x1 = np.asfarray(data.values.flatten(),float)
        x1= x1.tolist()

        header_count = len(data)
        x2 = [i for i in range(1, header_count+1)]


        temp_df = pd.DataFrame()
        temp_df.insert(loc=0, column='Num', value=x2)
        temp_df.insert(loc=1, column='Col_Data', value=x1)
        dd={"num" :x2 ,"col":x1}
        #for i in range (0,len(data)):
          #  dd[x2[i]]= x1[i]

        os.remove("files/data.csv")
        print("File Removed!")





        return dd


nn=do_local()
