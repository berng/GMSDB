import numpy as np
import time
from sklearn.mixture import GaussianMixture
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
# import autoscale_ogm as asogm
import matplotlib.pyplot as pp
import sys
from sklearn.preprocessing import StandardScaler


import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import dbscan

#import GMSDB as gmsdb


from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_blobs
import matplotlib.pyplot as pp
 
NF=2
# X1, y1 = make_blobs(n_samples=5000, centers=[(0,0)], n_features=NF,random_state=0)
# X1[:,0]*=10
# X1[:,1]+=0.07*X1[:,0]**2
# X2, y2 = make_blobs(n_samples=5000, centers=[(0,0)], n_features=NF,random_state=0)
# X2[:,0]*=10
# X2[:,1]-=0.07*X2[:,0]**2-50

# X=np.concatenate([X1,X2+20],axis=0)
# Y=np.concatenate([y1,y2+1],axis=0)
#X=X[:100]
#Y=Y[:100]

NF=2
X, Y = make_blobs(n_samples=10000, centers=5, n_features=NF,random_state=0,cluster_std=0.5)
Nn=300
Xn1=np.random.rand(Nn,1)*(X[:,0].max()-X[:,0].min())+X[:,0].min()
Xn2=np.random.rand(Nn,1)*(X[:,1].max()-X[:,1].min())+X[:,1].min()
Xn=np.concatenate([Xn1,Xn2],axis=1)
Yn=np.zeros(Nn)
Yn[:]=Y.max()+1
X=np.concatenate([X,Xn],axis=0)
Y=np.concatenate([Y,Yn],axis=0)
if __name__=='__main__':
 pp.scatter(X[:,0],X[:,1],c=Y)
 pp.show()
