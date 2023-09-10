# GMSDB Clusterer

Run for simple test:

 python GMSDB.py


## Simple use (sklearn -like style), no speed improvement, standard significance level alpha=0.05. 
Maximal number of gaussian components=50:

from GMSDB import GMSDB

clf=GMSDB(n_components=50)

clf.fit(X)

Y=clf.predict(X)

## Complex use (with speed improvement for stages 1 and 3):

clf=GMSDB(min_components=2,step_components=100,n_components=900,rand_search=1000,rand_level=0.5)

## Complex use (with custom significance level alpha=0.15):

clf=GMSDB(n_components=50,alpha_stage2=0.15,alpha_stage4=0.15)

## Verbose use (show debug information):

clf=GMSDB(n_components=50,verbose=True)

# Install with pip
You could install it from PyPI:

pip install gmsdb

Import:

from gmsdb import GMSDB
