from numpy import *
import pandas as pd
import numpy as np
#Import dataset
dataSet1 = pd.read_csv('../Data/GSPC.csv', header=None)
#dataSet2 = pd.read_csv('../Data/GSPC2.csv', header=None)
dt = dataSet1.values[16:, 0:1]
adj_close_0 = dataSet1.values[16:, 5:6]
adj_close_1 = dataSet1.values[15:-1, 5:6]
adj_close_2 = dataSet1.values[14:-2, 5:6]
adj_close_4 = dataSet1.values[12:-4, 5:6]
adj_close_8 = dataSet1.values[8:-8, 5:6]
adj_close_16 = dataSet1.values[0:-16, 5:6]

label = adj_close_0 - adj_close_1

for i, lb in enumerate(label):
    if lb >= 0:
        label[i] = 1
    else:
        label[i] = 0


#print(adj_close_0)
#print(shape(adj_close_0))
#print(shape(adj_close_1))
#print(shape(adj_close_2))
#print(shape(adj_close_4))
#print(shape(adj_close_8))
#print(shape(adj_close_16))

#calc lat
'''lat_1 = abs(np.divide(adj_close_0, adj_close_1) - 1)
lat_2 = abs(np.divide(adj_close_0, adj_close_2) - 1)
lat_4 = abs(np.divide(adj_close_0, adj_close_4) - 1)
lat_8 = abs(np.divide(adj_close_0, adj_close_8) - 1)
lat_16 = abs(np.divide(adj_close_0, adj_close_16) - 1)

dataset_lat1 = pd.DataFrame(lat_1)
dataset_lat2 = pd.DataFrame(lat_2)
dataset_lat4 = pd.DataFrame(lat_4)
dataset_lat8 = pd.DataFrame(lat_8)
dataset_lat16 = pd.DataFrame(lat_16)
dataset_label = pd.DataFrame(label)

dataset_lat1.to_csv('../Data/lat_1.csv', sep=',', na_rep='', float_format='lat_1', columns=None, header=False, index=False)
dataset_lat2.to_csv('../Data/lat_2.csv', sep=',', na_rep='', float_format='lat_1', columns=None, header=False, index=False)
dataset_lat4.to_csv('../Data/lat_4.csv', sep=',', na_rep='', float_format='lat_1', columns=None, header=False, index=False)
dataset_lat8.to_csv('../Data/lat_8.csv', sep=',', na_rep='', float_format='lat_1', columns=None, header=False, index=False)
dataset_lat16.to_csv('../Data/lat_16.csv', sep=',', na_rep='', float_format='lat_1', columns=None, header=False, index=False)
dataset_label.to_csv('../Data/label.csv', sep=',', na_rep='', float_format='lat_1', columns=None, header=False, index=False)'''
#print(lat_1)
