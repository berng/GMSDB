# Made under GNU GPL v3.0
# by Oleg I.Berngardt, 2023
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as 
# published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. 
# If not, see <https://www.gnu.org/licenses/>. 
    
import sys
import numpy as np
import matplotlib.pyplot as pp
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import dbscan
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
# import mahalanobis as mh_dist

from mlxtend.plotting import plot_decision_regions

import scipy.stats as ss
from numpy.linalg import inv
from statsmodels.stats.multitest import multipletests

def getMahalanobisMatrix(X,gm,alpha=0.05,show=False,N=-1):
 percent=int(alpha*100)
# N=-1
 Y=gm.predict(X)
 cov=gm.covariances_
# print(cov.shape)
 DD=np.zeros((gm.n_components,gm.n_components))
 DN=np.zeros((gm.n_components,gm.n_components))
 for i in range(gm.n_components):
  for j in range(gm.n_components):
#   print('process ',i,j,end=' ')
   
   pd_ij = pairwise_distances(X[Y==i][:N],X[Y==j][:N],metric='mahalanobis',VI=inv(cov[j]))
   pd_ij=pd_ij.reshape(pd_ij.shape[0]*pd_ij.shape[1])
   pd_ji = pairwise_distances(X[Y==j][:N],X[Y==i][:N],metric='mahalanobis',VI=inv(cov[i]))
   pd_ji=pd_ji.reshape(pd_ji.shape[0]*pd_ji.shape[1])
   pd_ii = pairwise_distances(X[Y==i][:N],X[Y==i][:N],metric='mahalanobis',VI=inv(cov[i]))
   pd_ii=pd_ii.reshape(pd_ii.shape[0]*pd_ii.shape[1])
   pd_jj = pairwise_distances(X[Y==j][:N],X[Y==j][:N],metric='mahalanobis',VI=inv(cov[j]))
   pd_jj=pd_jj.reshape(pd_jj.shape[0]*pd_jj.shape[1])
   cond=np.where(np.isnan(pd_ii),False,True)
   pd_ii=pd_ii[cond]
   cond=np.where(np.isnan(pd_jj),False,True)
   pd_jj=pd_jj[cond]
   cond=np.where(np.isnan(pd_ij),False,True)
   pd_ij=pd_ij[cond]
   cond=np.where(np.isnan(pd_ji),False,True)
   pd_ji=pd_ji[cond]

   d_ii=np.median(pd_ii)
   d_jj=np.median(pd_jj)

   if pd_ii.shape[0]*pd_ij.shape[0]*pd_ji.shape[0]*pd_jj.shape[0]>0:
    d_ij=max(np.percentile(pd_ij,percent),np.percentile(pd_ji,percent))
   elif pd_ij.shape[0]*pd_ji.shape[0]==0:
    if pd_ij.shape[0]>pd_ji.shape[0]:
       d_ij=np.percentile(pd_ij,percent)
    elif pd_ji.shape[0]>pd_ij.shape[0]:
       d_ij=np.percentile(pd_ji,percent)
    else:
       d_ij=-1

   if d_ij>=0:
    DD[i,j]+=d_ij
    DN[i,j]+=1
   DD[i,i]+=d_ii
   DN[i,i]+=1
   DD[j,j]+=d_jj
   DN[j,j]+=1

   if show:
    pp.figure()
    fig,axs=pp.subplots(2,1)
    axs[0].hist(pd_ii[np.where(np.isnan(pd_ii),False,True)],bins=100,histtype='step',density=True,label='ii')
    axs[0].hist(pd_ij[np.where(np.isnan(pd_ij),False,True)],bins=100,histtype='step',density=True,label='ij') 
    axs[0].hist(pd_ji[np.where(np.isnan(pd_ji),False,True)],bins=100,histtype='step',density=True,label='ji') 
    axs[0].hist(pd_jj[np.where(np.isnan(pd_jj),False,True)],bins=100,histtype='step',density=True,label='jj')
    axs[0].legend()
    cond=np.where((Y==i) | (Y==j))
    axs[1].scatter(X[cond,0],X[cond,1],c=Y[cond])
    pp.savefig('img/'+str(i)+'-'+str(j)+'.jpg')
    pp.close()
 DD/=DN
 cond=np.where((np.isnan(DD) | np.isinf(DD)),True,False)
 notcond=np.where((np.isnan(DD) | np.isinf(DD)),False,True)
 max1=DD[notcond].max()
 DD[cond]=max1
 return DD



def ConfiDistance(MatrixClustersAll,gm=False,X=np.zeros(0),eps=1.,show=False,fname='tmp.png',\
           MaxTestSize=1000,border_percentile=0.001,alpha=0.05,beta=0.5,bh_mh_correct=False):
   PERECENTILE=50   # border for maximal distance, 50% = median
   BORDER=ss.norm().ppf(1.-alpha*beta)
#   print('BORDER:',BORDER)
#   quit()
   MC=MatrixClustersAll.copy()
#   if show:
#    print('MCA:\n',MatrixClustersAll)
   for i in range(MC.shape[0]):
    MC[i,i]=0.
   idx,labels_superclusters=dbscan(MC, metric='precomputed', eps=eps,min_samples=1)

   total_superclusters=len(set(list(labels_superclusters)))
   if(total_superclusters)<2:
    return -1e100,labels_superclusters   

# mean distance to closest cluster + 1/2 cluster size
   ISCmin=np.ones((total_superclusters,total_superclusters))*1e100
   for i_scl in range(total_superclusters):
    for j_scl in range(total_superclusters):
     index_i=labels_superclusters==i_scl
     index_j=labels_superclusters==j_scl
     MC=MatrixClustersAll.copy()
     np.fill_diagonal(MC,1e100)
     ISCmin[i_scl,j_scl]=(MC[index_i,:][:,index_j]).min()
    index_i=labels_superclusters==i_scl
#    ISCmin[i_scl,i_scl]=max((MatrixClustersAll[index_i,:][:,index_i]).min(),ISCmin[i_scl,:].min())
#   ISCmin[i_scl,i_scl]=min((MatrixClustersAll[index_i,:][:,index_i]).min(),ISCmin[i_scl,:].min())
    ISCmin[i_scl,i_scl]=np.diag(MatrixClustersAll)[index_i].min()
   if show:
    print('tot:',total_superclusters)

   ISCmin2=ISCmin/np.diag(ISCmin)
#   ISCmin2-=1.
   if not bh_mh_correct:
    np.fill_diagonal(ISCmin2,1e100)
    distA=ISCmin2.min(axis=1)
    dist=distA[distA>BORDER].shape[0]/distA.shape[0]
   else:
    pvals_arr=(1.-ss.norm.cdf(ISCmin2))*2.
    np.fill_diagonal(pvals_arr,1.0)
    rej,pvals_arr_corr,a,b=multipletests(pvals_arr.reshape(pvals_arr.shape[0]*pvals_arr.shape[1]),alpha=alpha,method='fdr_bh')
    pvals_arr_corr=pvals_arr_corr.reshape(pvals_arr.shape[0],pvals_arr.shape[1])
    np.fill_diagonal(pvals_arr_corr,0.0)
    distB=pvals_arr_corr.max(axis=1)
    dist=distB[distB<alpha].shape[0]/distB.shape[0]

   if show:
#    print('ISCmin:\n',np.round(ISCmin2,1))
    print('dist:\n',dist)
#   print('mydist:',dist)
   return dist,labels_superclusters
#   quit()

## Calc modified Silhouette
   b_i=np.ones(total_superclusters)*1e100
   a_i=np.ones(total_superclusters)*0
   for i_scl in range(total_superclusters):
    Sil_i=[]
    for j_scl in range(total_superclusters):
     if i_scl!=j_scl:
      if ISCmin[i_scl,j_scl]<b_i[i_scl]:
       b_i[i_scl]=ISCmin[i_scl,j_scl]
    a_i[i_scl]=ISCmin[i_scl,i_scl]

   if show:
    print('ai,bi:\n',a_i,'\n',b_i)
   sil_i=b_i/a_i			# modified silhouette (not normalized)
#   sil_i=(b_i-a_i)/np.maximum(b_i,a_i)	# original silhouette
   if show:
    print('======sil_i',sil_i,sil_i.min(),sil_i.max(),sil_i.std())

#   old_labels_superclusters[hash_list]=sil_i.min()

#   return sil_i.min(),labels_superclusters
#   return sil_i.min()*total_superclusters,labels_superclusters
   return sil_i.min(),labels_superclusters

class GMSDB():
 def __init__(self,min_components=2,step_components=1,n_components=2,verbose=False,show=False,metric='LH',\
               border_percentile=0.001,alpha_stage2=0.05,alpha_stage4=0.05,show_mah=False, \
               show_clusters=False,autostop=True,bh_mh_correct=False,rand_search=0,rand_level=0.5):
# ==== input parameters
# 1.Basic parameters:
  self.n_components=n_components               	# maximal number of gaussian clusters, the more - the slower the stage 1.
						# I use 100 for simple cases (dot examples), 
						# and 1000 with speed up search paramters (below: 2,100,1000,0.5) for complex cases (image examples)
  self.alpha_stage2=alpha_stage2		# significance level for stage 2. Recommended standard value 0.05
  self.alpha_stage4=alpha_stage4		# significance level for stage 4. Recommended standard value 0.05

# 2.Speed up search parameters:
  self.min_components=min_components           	# minimal number of gaussian clusters (stage 1), in most cases use 2
  self.step_components=step_components	       	# fast algorithm parameter for BIC search (stage 1). 
						# Recommended values: 100 (if n_components>=1000); 
                                                #                     20 if 100<n_components<1000; 
                                                #                     or 1 if n_components<50
  self.rand_search=rand_search			# set to 0 if do not want to speed up search at stage 3-4. 
						# Set to 1000 or more to speed up the search at stage 3-4. 
                                                # Not recommended to set more than n_components
  self.rand_level=rand_level			# Autostop random search when MC reaches this value (stage 3 speed optimization). 
						# Then algorithm uses sequental search. 
						# In most cases default value 0.5 is a good variant

# 3.Debug only keys
  self.SUBITT=1					# used, when one wants to make several itterations with the same eps. Used with show flag for studies of algorithm
  self.verbose=verbose			       	# use for debug outputs
  self.show=show				# use for producing debug images
  self.show_mah=show_mah			# use for debug Mahalanobis distance
  self.show_clusters=show_clusters		# use for producing debug images to show clusters
  self.bh_mh_correct=bh_mh_correct		# use for multiple hypothesis correcting. Not recommended to use
  self.autostop=autostop			# use for stopping when reaching MC==1. Set to False for debug only purposes

# 4.not used keys
  self.metric=metric				# not used, historical code
  self.border_percentile=border_percentile	# not used, historical code
  self.ZeroDistance=0.001			# not used, historical code

# ==== Temporary paramters of the algorithm
  self.bics=np.ones(n_components)*1e100
  self.gms={}
  self.lower_bound_=0				# for suport sklearn, always 0 

# ==== Output parameters
  self.ss=False					# Trained input standard scaler
  self.gm=False					# BIC-optimal gaussian classifier
  self.labels=[]				# found correspondence between gaussian clusters and supercluster numbers
  self.best_cluster_num=0			# found number of superclusters

 def getMatrix(self,X,clusters,fname=''):
  return getMahalanobisMatrix(X,self.gm,alpha=self.alpha_stage2,show=self.show_mah)


 def getOptSuperclusters(self,Matrix,gm,X,Y,eps_np_mid):
  self.labels=np.array(list(set(list(Y))))

  checked_indexes={}
  for i in range(1,self.opt_bics+1):
   checked_indexes[i]=0


  cdist_opt=-1e100
  eps_opt=-1
  eps_history={}
  optimals=[]
  tot_cl=100
#  self.rand_search=10
#  self.rand_search=False
  search_idx=np.array(range(eps_np_mid.shape[0]))
  if self.rand_search>0:
   np.random.shuffle(search_idx)
  search_idx=list(search_idx)
  eps_supermin=1e100
  eps_supermax=-1e100
  eps_checked={}
  
  if self.rand_search>0:
   np.random.shuffle(search_idx)
   for eps_itt in search_idx[:self.rand_search]:
     eps=eps_np_mid[eps_itt]
     if eps <eps_supermax:
       print('.',end='',flush=True)
       continue
     if eps >eps_supermin:
       print('.',end='',flush=True)
       continue

     MatrixTmp=Matrix.copy() #self.getMatrix(X,clusters)
     cdist,labels=ConfiDistance(MatrixTmp,self.gm,X,eps=eps,show=self.verbose,fname='img/'+str(eps_itt)+'.png',\
                       border_percentile=self.border_percentile,alpha=self.alpha_stage4,bh_mh_correct=self.bh_mh_correct)
     if 1>=cdist>=0.:
       eps_checked[eps_itt]=cdist
     elif cdist<=0.:
       eps_checked[eps_itt]=1.

     if cdist>=1.0  or len(set(list(labels)))==1 or  cdist==-1e100:
       if self.verbose:
        print('cmp 2 eps_smin,eps:',eps_supermin,eps,flush=True)
       if eps<eps_supermin:
        eps_supermin=eps
       if self.verbose:
        print('changed 2 eps_smax,eps_min limits:',eps_supermax,eps_supermin,flush=True)

     if 0.<=cdist<1.:
       if self.verbose:
        print('cmp 1 eps_smax,eps:',eps_supermax,eps,flush=True)
       if eps>eps_supermax:
        eps_supermax=eps
       if self.verbose:
        print('changed 1 eps_smax,eps_min limits:',eps_supermax,eps_supermin,flush=True)
     if eps_supermax>0 and eps_supermin<1e100 and (1.>cdist>self.rand_level):
        break

  if self.verbose:
   print('will search over eps_smax,eps_min limits:',eps_supermax,eps_supermin,flush=True)
   for eid in eps_checked.keys():
    print(eid,eps_np_mid[eid],eps_checked[eid],file=sys.stderr)
#  quit()
# /Random search

  search_idx=np.array(range(eps_np_mid.shape[0]))
  search_idx=list(search_idx)

  for eps_itt in search_idx:
    eps=eps_np_mid[eps_itt]
    # skip variants outside current limits
    if eps<=eps_supermax:
      continue
    if eps>eps_supermin:
      continue

#    eps_checked[eps_itt]=True
    for subitt in range(self.SUBITT):
     MatrixTmp=Matrix.copy() #self.getMatrix(X,clusters)
     cdist,labels=ConfiDistance(MatrixTmp,self.gm,X,eps=eps,show=self.verbose,fname='img/'+str(eps_itt)+'.png',\
                       border_percentile=self.border_percentile,alpha=self.alpha_stage4,bh_mh_correct=self.bh_mh_correct)

     if self.show_clusters:
      print('showing cluster...', eps_itt)
      pp.figure()
      pp.scatter(X[:,0],X[:,1],c=labels[Y])
      pp.savefig('img/cl-'+str(eps_itt)+'.jpg')
      pp.close()

     tot_cl=len(set(list(labels)))

#     if self.verbose:
#      print('before inc:',len(checked_indexes),checked_indexes)
     if tot_cl in checked_indexes:
      checked_indexes[tot_cl]+=1
      if tot_cl>1:
       checked_indexes[1]=0

#     if self.verbose:
#      print('after inc:',len(checked_indexes),checked_indexes)
     eps_history[eps]={'classes':tot_cl,'labels':labels,'cdist':cdist}

#     quit()
#     if self.save_json:
     if self.verbose:
      with open('checked_indexes.json','wb') as f:
       pickle.dump(checked_indexes,f)
       f.close()
      with open('eps_history.json','wb') as f:
       pickle.dump(eps_history,f)
       f.close()

     if self.verbose:
      print('check',eps_itt,'/',eps_np_mid.shape[0],' eps:',eps,'cdist',cdist,'labels:',len(set(list(labels))),'eps lim:',eps_supermax,eps_supermin)

     if cdist>cdist_opt and cdist_opt<=1.:
      cdist_opt=cdist
      eps_opt=eps
      self.best_cluster_num=tot_cl
      if self.verbose:
       print('it',eps_itt,'cdist',cdist,'labels:',len(set(list(labels))))
       optimals.append({'eps':eps,'classes':tot_cl,'labels':labels,'cdist':cdist})
       with open('optimals.json','wb') as f:
        pickle.dump(optimals,f)
        f.close()
       print('============== optimal!!! ===============',flush=True)
      self.labels=labels

     if len(set(list(labels)))<=1  and checked_indexes[1]>10:
      if self.verbose:
       print('============== no more itterations needed - only 1 class ! ==========',flush=True)
      return self.best_cluster_num,self.labels

     if  cdist>=1.0 and self.autostop:
      if self.verbose:
       print('============== no more itterations needed - absolute optimum reached! ==========',flush=True)
      return self.best_cluster_num,self.labels
 
     if len(set(list(labels)))>self.opt_bics:
      if self.verbose:
       print('it',eps_itt,'cdist',cdist,'labels:',len(set(list(labels))))
       print('============== unexpected break!!! ===============',flush=True)
      return self.best_cluster_num,self.labels


    if  len(set(list(labels)))<=1 and checked_indexes[1]>10:
     if self.verbose:
       print('============== no more itterations needed - only 1 class ! ==========',flush=True)
     return self.best_cluster_num,self.labels
# final return

  if self.verbose:
   print('============== Unpredicted return ! ==========',cdist_opt,self.best_cluster_num,self.labels,eps_supermax,eps_supermin,flush=True)
#   for eid in eps_checked.keys():
#    print(eid,eps_np_mid[eid],eps_checked[eid],file=sys.stderr)
  return self.best_cluster_num,self.labels


 def fit(self,X0,show_closest=False,plot_decisions=False):
  ss=StandardScaler()
  X=ss.fit_transform(X0)
  self.ss=ss
  best_bic=1e100
  best_C=-1
  mymin=self.min_components
  mymax=self.n_components
  mystep=self.step_components
  for itt in range(10):
   for C in range(mymin,mymax,mystep):

    if not C in self.gms:
     gm=GaussianMixture(n_components=C)
     gm.fit(X)
    else:
     gm=self.gms[C]

    yp=gm.predict(X)
    self.bics[C]=gm.bic(X)
    self.gms[C]=gm

    if self.bics[C]<best_bic:
      best_bic=self.bics[C]
      best_C=C
      self.gm=gm
      self.yp=yp

      if plot_decisions:
       pp.figure()
#       pp.scatter(X[:,0],X[:,1],c=yp)
       plot_decision_regions(X,yp,clf=self.gm)
       pp.savefig('bicfile.png')
       pp.close()

    if self.verbose:
     print("C:",C,'bic:',self.bics[C],flush=True)
   if(mystep>1):
    mymin=best_C-mystep-1
    mymax=best_C+mystep
    if mystep>5:
     mystep=mystep//5
    else:
     mystep=1
    if self.verbose:
     print('new itt:',mymin,mymax,mystep)
   else:
    break
  self.opt_bics=np.argmin(self.bics[self.min_components:])+self.min_components
  if self.verbose:
    print('optbic:',self.opt_bics,flush=True)
#  for C in range(int(self.opt_bics*1.5),self.n_components,1):
#   if self.bics[C]<1e100:
#      self.gm=self.gms[C]
#      self.opt_bics=C
#      print('increase bics to ',self.opt_bics)
#      break

#  quit()

  Y=self.gm.predict(X)
#  pp.scatter(X[:,0],X[:,1],c=Y)
#  pp.show()

  clusters=set(list(Y))
  Matrix=self.getMatrix(X,clusters)
#  show_closest=False
  if show_closest:
   for i in range(Matrix.shape[0]): 
    MC=Matrix.copy()
    MC[i,i]=1e100
    iclosest=np.argmin(MC[i,:])
    Yc=Y.copy()
    Yc[Yc==i]=-1
    Yc[Yc==iclosest]=-2
    MC[i,iclosest]=1e100
    iclosest2=np.argmin(MC[i,:])
    Yc[Yc==iclosest2]=-3
    Yc[Yc>=0]=0
    pp.figure()
    for c in set(list(Yc)):
     pp.scatter(X[Yc==c,0],X[Yc==c,1],label=str(c))
    pp.legend()
    pp.savefig('img/f-'+str(i)+'.jpg')
    pp.close()
   quit()
  if self.verbose:
   print('prep matr:\n',np.round(Matrix,2))    

  eps_vals=set(list(Matrix.reshape(Matrix.shape[0]*Matrix.shape[1])))
  eps_np=np.array(sorted(eps_vals))
  eps_np_mid=(eps_np[1:]+eps_np[:-1])/2.

  if self.verbose:
   print('unique matrix vals:',eps_np_mid,eps_np_mid.shape[0])

#  for itt in range(10):
  best_cl_num,superlabels=self.getOptSuperclusters(Matrix,self.gm,X,Y,eps_np_mid)
#   print('itt,best_cl_num,superclusters:',itt,best_cl_num,superlabels,flush=True)



#    break




 def predict(self,X0):
  X=self.ss.transform(X0)
  Y=self.gm.predict(X)
  return self.labels[Y]

 def predict_proba(self,X0):
  X=self.ss.transform(X0)
  Y=self.gm.predict_proba(X)
  superclusters=set(list(self.labels))
  res=np.zeros((Y.shape[0],len(superclusters)))
  for scl in superclusters:
   res[:,scl]=Y[:,self.labels==scl].sum(axis=1)
#  res/=np.expand_dims(res.sum(axis=1),-1)
  return res

 def fit_predict(self,X):
  self.fit(X)
  return self.predict(X)

 def aic(self,X):
  return 0

 def bic(self,X):
  return 0

if __name__=='__main__':
 from mlxtend.plotting import plot_decision_regions
 from sklearn.datasets import make_blobs
 import matplotlib.pyplot as pp

 print('wait a couple minutes...')

# test dataset
 NF=2
 X1=np.zeros((5000,2))
 y1=np.zeros(5000)
 r=np.random.rand(5000)*2.7
 phi=np.random.rand(5000)*6.28
 X1[:,0]=r*np.cos(phi)
 X1[:,1]=r*np.sin(phi)
 X1[:,0]*=10
 X1[:,1]+=0.07*X1[:,0]**2
 X2=np.zeros((5000,2))
 y2=np.zeros(5000)
 r=np.random.rand(5000)*2.7
 phi=np.random.rand(5000)*6.28
 X2[:,0]=r*np.cos(phi)
 X2[:,1]=r*np.sin(phi)
 X2[:,0]*=10
 X2[:,1]-=0.07*X2[:,0]**2-50
 X3=X1.copy()
 X3[:,0]+=65
 X=np.concatenate([X1,X2+30,X3],axis=0)
 Y=np.concatenate([y1,y2+1,y1+2],axis=0)



 NCM=35
 clf=GMSDB(n_components=NCM)
 clf.fit(X)
 print('sclusters:',clf.labels)
 y=clf.gm.predict(clf.ss.transform(X))
 Yp=clf.predict(X)
 pp.figure()
 fig,axs=pp.subplots(1,3,figsize=(13,4))
 axs[0].set_title('Source')
 axs[0].scatter(X[:,0],X[:,1],s=1,label='s')
 axs[1].set_title('GM BIC clusters')
 axs[1].scatter(X[:,0],X[:,1],c=y,s=1,label='gm')
 axs[2].set_title('superclusters')
 axs[2].scatter(X[:,0],X[:,1],c=Yp,s=1,label='sc')
 pp.savefig('res.png')
 pp.close()

 Yp=clf.predict_proba(X)
 print(Yp[:10,:])
 quit()

