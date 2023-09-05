import sys
sys.path.append('../')
from GMSDB import GMSDB

if __name__=='__main__':
 from mlxtend.plotting import plot_decision_regions
 from sklearn.datasets import make_blobs
 import matplotlib.pyplot as pp
 import data as data 
 X=data.X
 Y=data.Y
 NCM=35
 clf=GMSDB(n_components=NCM,alpha_stage2=0.05,alpha_stage4=0.05,verbose=True,rand_search=10,rand_level=0.5)
# clf.fit(X,plot_decisions=True)
 clf.fit(X)
 print('sclusters:',clf.labels)
 y=clf.gm.predict(clf.ss.transform(X))
 Yp=clf.predict(X)
 pp.figure()
 fig,axs=pp.subplots(1,2,figsize=(10,4))
 axs[0].set_title('orig')
 axs[0].scatter(X[:,0],X[:,1],c=y,s=1,label='o')
 axs[1].set_title('superclusters')
 axs[1].scatter(X[:,0],X[:,1],c=Yp,s=1,label='sc')

 pp.savefig('res.png')
 pp.close()
# pp.figure()
# plot_decision_regions(X, Yp, clf=clf, legend=0)
# pp.title('classes:'+str(clf.opt_bics))
# pp.savefig('dec.png')

 Yp=clf.predict_proba(X)
 print(Yp[:10,:])
 quit()

