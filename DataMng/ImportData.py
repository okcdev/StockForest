from numpy import *
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import pandas as pd

plt.rcParams['figure.figsize'] = (12, 6)
style.use('ggplot')

#Import DataSet
data = pd.read_csv('../Data/GSPC.csv', header=None)
X = data.values[:, :6]
#print(X)
dataSet = []
Label = []
#Extractor data
for i in range(len(X)):

    #date
    dt = X[i][0]
    
    #Open
    open_1 = round(float(X[i][1]),5)
    
    #lat_1
    lat_1 = abs(float(X[i][5])/float(X[i-1][5])-1)
    lat_1 = round(lat_1,5)
    #print(lat_1)

    #lat_2
    lat_2 = abs(float(X[i][5])/float(X[i-2][5])-1)
    lat_2 = round(lat_2,5)

    #lat_4
    lat_4 = abs(float(X[i][5])/float(X[i-4][5])-1)
    lat_4 = round(lat_2,5)

    #lat_8
    lat_8 = abs(float(X[i][5])/float(X[i-8][5])-1)
    lat_8 = round(lat_8,5)

    #lat_16
    lat_16 = abs(float(X[i][5])/float(X[i-8][5])-1)
    lat_16 = round(lat_16,5)
    
    #Label
    if round(float(X[i][5])-float(X[i-1][5]),2)>0:
        label = 1
    else:
        label = 0
        
    ds = [lat_1, lat_2, lat_4, lat_8, lat_16]
    dataSet.append(ds)
    Label.append(label)
dataSet=np.array(dataSet)'dataSet = mat(dataSet)
Label=np.array(Label)
print(dataSet)
print(Label)

def draw_svm(X, y, C=1.0):
    # Plotting the Points
    plt.scatter(X[:,0], X[:,1], c=y)
    
    # The SVM Model with given C parameter
    clf = LogisticRegression()
    #clf = SVC(kernel='linear', C=C)
    clf_fit = clf.fit(X, y)
    
    #linit of the axes
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    #Creating the meshgrid
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1],200)

    plt.show()
    
    # Returns the classifier
    return clf_fit

#draw_cavns
def draw_cavns(X, y, C=1.0):
    #Plotting the Points
    plt.scatter(X[:,0], X[:,1], c=y)

    # The Logist Model
    #clf = LogisticRegression(C,penalty = 'l1', tol=0.01)
    clf =SVC(kernel='linear', C=C)
    clf_fit = clf.fit(X, y)

    #linit of the axes
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    #Creating the meshgrid
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1],200)

    YY,XX = np.meshgrid(yy,xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    #Plotting the boundary
    ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1],
               alpha=0.5, linestyles=['--','-','--'])
    ax.scatter(clf.support_vectors_[:, 0],
               clf.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none')
    plt.show()
    return clf_fit

#cal_arr = draw_svm(dataSet, Label, 1)
#print(cal_arr.score(dataSet, Label))
#print(cal_arr.coef_)
#pred = cal_arr.predict([(0.00319,0.00485,0.00485, 0.00452, 0.00452)])
#print(pred)


