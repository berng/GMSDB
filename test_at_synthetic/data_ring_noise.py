import numpy as np
import time
from sklearn.mixture import GaussianMixture
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as pp
# import autoscale_ogm as asogm
import sys
from sklearn.preprocessing import StandardScaler


import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import dbscan

#import GMSDB as gmsdb


from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_blobs
import matplotlib.pyplot as pp
 
NN=1000
r=np.random.rand(NN)*20+50
r2=np.random.rand(NN)*20
phi=np.random.rand(NN)*6.28
phi2=np.random.rand(NN)*6.28
X1=np.zeros((NN,2))
Y1=np.zeros(NN)
X1[:,0]=r*np.cos(phi)
X1[:,1]=r*np.sin(phi)
Y1[:]=1

X2=np.zeros((NN,2))
Y2=np.zeros(NN)
X2[:,0]=r2*np.cos(phi2)
X2[:,1]=r2*np.sin(phi2)
Y2[:]=0

X=np.concatenate([X1,X2],axis=0)
Y=np.concatenate([Y1,Y2],axis=0)

#pp.scatter(X[:,0],X[:,1])
#pp.show()
#quit()

Nn=300
Xn1=np.random.rand(Nn,1)*(X[:,0].max()-X[:,0].min())+X[:,0].min()
Xn2=np.random.rand(Nn,1)*(X[:,1].max()-X[:,1].min())+X[:,1].min()
Xn=np.concatenate([Xn1,Xn2],axis=1)
Yn=np.zeros(Nn)
Yn[:]=Y.max()+1
X=np.concatenate([X,Xn],axis=0)
Y=np.concatenate([Y,Yn],axis=0)
