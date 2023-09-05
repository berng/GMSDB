#!python
import matplotlib.pyplot as pp
import sys
import numpy as np
from PIL import Image
sys.path.append('../')
import GMSDB as gmsdb


print ("usage: "+sys.argv[0]+" greyscaleimage256x256.png alpha")
im = Image.open(sys.argv[1])
alpha=float(sys.argv[2])
print("Attantion!!! Very slow, can take an hour")

im_np=np.array(im).astype(float)
print(im_np.shape)
im_np=(im_np-im_np.min())/(im_np.max()-im_np.min())
print('d:',im_np.min(),im_np.max())
d=[]
for x in range(0,im_np.shape[1],1):
 for y in range(0,im_np.shape[0],1):

  prob=im_np[y,x]
  d.append([x,y,prob])
d=np.array(d)


clf=gmsdb.GMSDB(min_components=500,step_components=20,n_components=600,verbose=True,alpha_stage2=alpha,alpha_stage4=alpha,rand_search=1000,rand_level=0.8)
clf.fit(d)
yp=clf.predict(d)
print('lables:',clf.labels)
fig,axs=pp.subplots(2,1,figsize=(8,12))
pl=np.array(im).astype(float)
axs[0].scatter(d[:,0],-d[:,1],c=d[:,2],s=1,alpha=0.5)
axs[1].scatter(d[:,0],-d[:,1],c=yp,s=1,alpha=0.5)
pp.savefig('gmsdb-'+sys.argv[1]+'.png')
print('done')
