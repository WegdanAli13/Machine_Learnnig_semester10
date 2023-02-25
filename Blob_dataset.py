import pandas
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn.datasets import make_blobs
from plotnine import *  
from mpl_toolkits import mplot3d
# StandardScaler is a function to normalize the data 
# You may also check MinMaxScaler and MaxAbsScaler 
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import DBSCAN


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")

n_bins = 6  
centers = [(-3, -3), (0, 0), (5,2.5),(-1, 4), (4, 6), (9,7)]
Multi_blob_Data, y = make_blobs(n_samples=[100,150, 300, 400,300, 200], n_features=2, cluster_std=[1.3,0.6, 1.2, 1.7,0.9,1.7],
                  centers=centers, shuffle=False, random_state=42)

#print(Multi_blob_Data)

#KMeans : 

# Elbow Method

inertias = []
for i in range(1,101):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(list(Multi_blob_Data))
    inertias.append(kmeans.inertia_)

plt.plot(range(1,101), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#---------------------------------------------------------------

#Clustring

kmeans = KMeans(n_clusters=6)
kmeans.fit(Multi_blob_Data)


plt.scatter(Multi_blob_Data[:,0],Multi_blob_Data[:,1],c=kmeans.labels_)
plt.show()

#---------------------------------------------------------------

# Hierarchal : 

# average 
plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(Multi_blob_Data, method ='average')))
plt.show()

ac2 = AgglomerativeClustering(n_clusters = 2,affinity="cosine",linkage="average")
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(Multi_blob_Data[:,0], Multi_blob_Data[:,1],
           c = ac2.fit_predict(Multi_blob_Data), cmap ='rainbow')
plt.show()

ac2 = AgglomerativeClustering(n_clusters = 2,affinity="euclidean",linkage="average")
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(Multi_blob_Data[:,0], Multi_blob_Data[:,1],
           c = ac2.fit_predict(Multi_blob_Data), cmap ='rainbow')
plt.show()



# #---------------------------------------------------------------

#single
plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(Multi_blob_Data, method ='single')))
plt.show()
ac2 = AgglomerativeClustering(n_clusters = 2,affinity="cosine",linkage="single")
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(Multi_blob_Data[:,0], Multi_blob_Data[:,1],
           c = ac2.fit_predict(Multi_blob_Data), cmap ='rainbow')
plt.show()

ac2 = AgglomerativeClustering(n_clusters = 2,affinity="euclidean",linkage="single")
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(Multi_blob_Data[:,0], Multi_blob_Data[:,1],
           c = ac2.fit_predict(Multi_blob_Data), cmap ='rainbow')
plt.show()

# #---------------------------------------------------------------

#manhattan
ac2 = AgglomerativeClustering(n_clusters = 2,affinity="manhattan",linkage="single")
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(Multi_blob_Data[:,0], Multi_blob_Data[:,1],
           c = ac2.fit_predict(Multi_blob_Data), cmap ='rainbow')
plt.show()

#---------------------------------------------------------------

# DBScan : 

xs = (x * 0.1 for x in range(1, 29))
ys = (y*1 for y in range(5, 30,5))
for i, j in zip(xs,ys):
 dbscan = DBSCAN(eps = i, min_samples = j).fit(Multi_blob_Data) # fitting the model
 labels = dbscan.labels_ # getting the labels
 plt.scatter(Multi_blob_Data[:, 0], Multi_blob_Data[:,1], c = labels, cmap= "rainbow") # plotting the clusters
 plt.show() # showing the plot

 #---------------------------------------------------------------

#Gaussian Mixture : 
