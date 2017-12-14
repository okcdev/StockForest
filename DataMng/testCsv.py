import numpy as np 
import pandas as pd
from pandas import *
from matplotlib import style
from sklearn.svm import SVC
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)
style.use('ggplot')

# Import Dataset
'''data = pd.read_csv('D:\APPS_WORK\PythonAPP\StockForest\Data\GSPC.csv', header=None)
X = data.values[:, 3:5]
y = data.values[:, 2]
print(y)
dataSet = []
for i in range(len(y)):
    print(y[i])
    dataSet.append(y[i])
print(dataSet)'''

'''from pandas import *  
from random import *  
df = DataFrame(columns=('lib', 'qty1', 'qty2'))#生成空的pandas表  
for i in range(5):#插入一行<span id="transmark" style="display: none; width: 0px; height: 0px;"></span>  
    df.loc[i] = [randint(-1,1) for n in range(3)]  
print (df)'''


# Import Dataset
data = pd.read_csv('D:\APPS_WORK\PythonAPP\StockForest\Data\GSPC.csv', header=None)
# x_arr = np.array()
data1 = []
X = data.values[:,3:5]
print(X)
i = 0
for x in X:
    if i == 0:
        x1 = x
    else:
        x2 = x
        x_x = x2-x1
        data1.append(x_x)
        x1 = x2
    i += 1
    # print(x)
x_arr = np.array(data1)
#y = data.values[:, 2]
# print(type(X))
print(x_arr)
