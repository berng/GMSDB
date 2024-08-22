# GMSDB Clusterer

Run for simple test:
```shell
python GMSDB.py
```

## Simple use (sklearn -like style), no speed improvement, default significance level alpha=0.1. 
Maximal number of gaussian components=50:

```python
from sklearn.datasets import make_blobs
from GMSDB import GMSDB

S,N=1000,5
X,Y=make_blobs(n_samples=S, n_features=2, centers=N, cluster_std=0.3, random_state=42)

clf=GMSDB(n_components=50)
clf.fit(X)
Y=clf.predict(X)
print(clf.get_centroids())
```

## Complex use (with speed improvement for stages 1 and 3):

```python
clf = GMSDB(min_components=2,step_components=100,n_components=900,rand_search=1000,rand_level=0.5)
```

## Complex use (with custom significance level alpha=0.15):

```python
clf=GMSDB(n_components=50,alpha_stage2=0.15,alpha_stage4=0.15)
```

## Verbose use (show debug information):
```python
clf=GMSDB(n_components=50,verbose=True)
```

# Install with pip
You could install it from PyPI:
```shell
pip install gmsdb==2.1
```

Import:
```python
from gmsdb import GMSDB
```

# Paper:
https://arxiv.org/pdf/2309.02623v2.pdf
