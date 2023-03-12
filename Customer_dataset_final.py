#This fileisorganised with the sequence for eachclustering technique , please uncomment the required technique needed before run

import pandas
import sys
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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import scipy.cluster.hierarchy as shc
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.decomposition import PCA
import plotly.express as px
#np.set_printoptions(threshold=sys.maxsize)



customer = pd.read_csv("/Users/wegdan/Desktop/Semester 10 🥳/Machine Learning/Assignment 1/Customer data.csv")
Customer_Data=np.vstack((customer.Sex,customer.Maritalstatus,customer.Age,customer.Education,customer.Income,customer.Occupation,customer.Settlementsize)).T

#--------------------------------------------------------------- 

# Data preparing : normalization and pca

Customer_Data=preprocessing.normalize(Customer_Data,axis=0)
pca = PCA(n_components=3)
Customer_Data = pca.fit_transform(Customer_Data)

#print(Customer_Data)

#--------------------------------------------------------------- 

#KMeans : 

# Elbow Method

# inertias = []
# for i in range(30,50):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(list(Customer_Data))
#     inertias.append(kmeans.inertia_)

# plt.plot(range(30,50), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()

#---------------------------------------------------------------

# silhouettte score

# # cosine distance 
# # The best silhoettte score is found at n=41 but n is high with random axis=0 (without it keeps changing) ss=0.8031
# score=[]

# for n in range(30,70,1):
 
#  kmeansc = KMeans(n_clusters=n,random_state = 0)
#  y_kmeans=kmeansc.fit_predict(Customer_Data)
#  score.append(silhouette_score(Customer_Data, kmeansc.labels_, metric='cosine'))
#  #print('Silhouetter Score: %.3f' % silhouette_score(Customer_Data, kmeans.labels_, metric='cosine') + " no of clusters= " + str(n))
# plt.title("silhouette score for range from 30 to 70 clusters in cosine distance  " )
# plt.plot(range(30,70),score,marker='o')
# plt.show()

# #---------------------------------------------------------------
# # euclidean distance 
# # The best silhoettte score is found at n=41 but n is high with random axis=0 (without it keeps changing) ss=0.7847

# score=[]
# for n in range(30,70,1):
 
#  kmeanse = KMeans(n_clusters=n,random_state = 0)
#  y_kmeans=kmeanse.fit_predict(Customer_Data)
#  score.append(silhouette_score(Customer_Data, kmeanse.labels_, metric='euclidean'))
#  #print('Silhouetter Score: %.3f' % silhouette_score(Customer_Data, kmeans.labels_, metric='euclidean') + " no of clusters= " + str(n))
#  plt.title("silhouette score for range from 30 to 70 clusters in euclidean distance  " )
# plt.plot(range(30,70),score,marker='o')
# plt.show()

# #---------------------------------------------------------------
# manhattan distance 
# The best silhoettte score is found at n=41 but n is high with random axis=0 (without it keeps changing) ss=0.7973

# score=[]
# for n in range(30,70,1):
 
#  kmeansm = KMeans(n_clusters=n,random_state = 0)
#  y_kmeans=kmeansm.fit_predict(Customer_Data)
#  score.append(silhouette_score(Customer_Data, kmeansm.labels_, metric='manhattan'))
#  #print('Silhouetter Score: %.3f' % silhouette_score(Customer_Data, kmeans.labels_, metric='manhattan') + " no of clusters= " + str(n))
#  plt.title("silhouette score for range from 30 to 70 clusters in manhattan distance  " )
# plt.plot(range(30,70),score,marker='o')
# plt.show()

#---------------------------------------------------------------
#clustering 

# kmeans = KMeans(n_clusters=41,random_state=0)
# y_kmeans=kmeans.fit_predict(Customer_Data)
# visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# visualizer.fit(Customer_Data)
# visualizer.show()
# ax = plt.axes(projection ="3d")
# for i in range(0,41):
#  ax.scatter3D(Customer_Data[y_kmeans==i,0], Customer_Data[y_kmeans==i,1],Customer_Data[y_kmeans==i,2],cmap='rainbow')
# plt.show()

#---------------------------------------------------------------

#Hierarchal : 

# scoreca=[]
# scorecs=[]

# scoreea=[]
# scorees=[]

# scorema=[]
# scorems=[]

# ax  = plt.axes(projection ="3d")

#---------------------------------------------------------------
#silhouette score


# for n in range(35,50,1):
#  ac1 = AgglomerativeClustering(n_clusters = n,affinity="cosine",linkage="average")
#  ac2 = AgglomerativeClustering(n_clusters = n,affinity="euclidean",linkage="average")
#  ac3 = AgglomerativeClustering(n_clusters = n,affinity="manhattan",linkage="average")    
#  ac4 = AgglomerativeClustering(n_clusters = n,affinity="cosine",linkage="single")
#  ac5 = AgglomerativeClustering(n_clusters = n,affinity="euclidean",linkage="single")
#  ac6 = AgglomerativeClustering(n_clusters = n,affinity="manhattan",linkage="single") 
 
#  y1=ac1.fit_predict(Customer_Data)
#  y2=ac2.fit_predict(Customer_Data)
#  y3=ac3.fit_predict(Customer_Data)
#  y4=ac4.fit_predict(Customer_Data)
#  y5=ac5.fit_predict(Customer_Data)
#  y6=ac6.fit_predict(Customer_Data)


 
#  scoreca.append(silhouette_score(Customer_Data, ac1.labels_ , metric='cosine'))
#  scorecs.append(silhouette_score(Customer_Data, ac4.labels_, metric='cosine'))

#  scoreea.append(silhouette_score(Customer_Data, ac2.labels_, metric='euclidean'))
#  scorees.append(silhouette_score(Customer_Data, ac5.labels_, metric='euclidean'))

#  scorema.append(silhouette_score(Customer_Data, ac3.labels_, metric='manhattan'))
#  scorems.append(silhouette_score(Customer_Data, ac6.labels_, metric='manhattan'))


# #ss=0.87967
# plt.figure()
# plt.title("silhouette score for range from 30 to 50 clusters cosine average  " )
# plt.plot(range(35,50),scoreca,marker='o')

# #ss=85386
# plt.figure()
# plt.title("silhouette score for range from 30 to 50 clusters cosine single  " )
# plt.plot(range(35,50),scorecs,marker='o')

# #0.7809
# plt.figure()
# plt.title("silhouette score for range from 30 to 50 clusters euclidean average  " )
# plt.plot(range(35,50),scoreea,marker='o')

# #ss=0.7678
# plt.figure()
# plt.title("silhouette score for range from 30 to 50 clusters euclidean single  " )
# plt.plot(range(35,50),scorees,marker='o')

# #ss=0.7957
# plt.figure()
# plt.title("silhouette score for range from 30 to 50 clusters manhattan average " )
# plt.plot(range(35,50),scorema,marker='o')

# #ss=0.7584
# plt.figure()
# plt.title("silhouette score for range from 30 to 50 clusters manhattan single  " )
# plt.plot(range(35,50),scorems,marker='o')
# plt.show()

#  #print('Silhouetter Score: %.3f' % score + " cosine average no of clusters= " + str(n))
#  #print('Silhouetter Score: %.3f' % score + " cosine single  no of clusters= " + str(n))
#  #print('Silhouetter Score: %.3f' % score + " euclidean average no of clusters= " + str(n))  
#  #print('Silhouetter Score: %.3f' % score + " euclidean single  no of clusters= " + str(n))
#  #print('Silhouetter Score: %.3f' % score + " manhattan average no of clusters= " + str(n))
#  #print('Silhouetter Score: %.3f' % score + " manhattan single  no of clusters= " + str(n))

#---------------------------------------------------------------
# dendogram

# plt.figure()
# plt.title("  aff=cosine link=average ")
# Dendrogram = shc.dendrogram((shc.linkage(Customer_Data, method ='average',metric='cosine')))

# plt.figure()
# plt.title("  aff=euclidean link=average ")
# Dendrogram = shc.dendrogram((shc.linkage(Customer_Data, method ='average',metric='euclidean')))

# plt.figure()
# plt.title("  aff=manhattan link=average ")
# Dendrogram = shc.dendrogram((shc.linkage(Customer_Data, method ='average',metric='cityblock')))

# plt.figure()
# plt.title("  aff=cosine link=single ")
# Dendrogram = shc.dendrogram((shc.linkage(Customer_Data, method ='single',metric='cosine')))

# plt.figure()
# plt.title("  aff=euclidean link=single ")
# Dendrogram = shc.dendrogram((shc.linkage(Customer_Data, method ='single',metric='euclidean')))

# plt.figure()
# plt.title("  aff=manhattan link=single ")
# Dendrogram = shc.dendrogram((shc.linkage(Customer_Data, method ='single',metric='cityblock')))

# plt.show()

#---------------------------------------------------------------
# clustering

# for cosine single n = 41
# ac4 = AgglomerativeClustering(n_clusters = 41,affinity="cosine",linkage="single")
# plt.title("cosine single  " )
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = ac4.fit_predict(Customer_Data), cmap ='rainbow')
# plt.show()

#for cosine average n = 47
# ac1 = AgglomerativeClustering(n_clusters = 47,affinity="cosine",linkage="average")
# plt.title("cosine average  " )
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = ac1.fit_predict(Customer_Data), cmap ='rainbow')
# plt.show()


# for euclidean single n = 49
# ac5 = AgglomerativeClustering(n_clusters = 49,affinity="euclidean",linkage="single")
# plt.title("euclidean single  " )
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = ac5.fit_predict(Customer_Data), cmap ='rainbow')
# plt.show()

# for euclidean average n = 49
# ac2 = AgglomerativeClustering(n_clusters = 49,affinity="euclidean",linkage="average")
# plt.title("euclidean average  " )
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = ac2.fit_predict(Customer_Data), cmap ='rainbow')
# plt.show()

# for manhattan single n = 49
# ac6 = AgglomerativeClustering(n_clusters = 49,affinity="manhattan",linkage="single")
# plt.title("manhattan single  " )
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = ac6.fit_predict(Customer_Data), cmap ='rainbow')
# plt.show()

# for manhattan average n = 48
# ac3 = AgglomerativeClustering(n_clusters = 48,affinity="manhattan",linkage="average")
# plt.title("manhattan average  " )
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = ac3.fit_predict(Customer_Data), cmap ='rainbow')
# plt.show()

#---------------------------------------------------------------

# DBScan : 0.005 30 ss=0.5773
ax  = plt.axes(projection ="3d")

# for i in np.arange (0.001,0.02,0.001):
 
#  score=[]

#  for j in range (30,50,1):
#   dbscan = DBSCAN(eps = i, min_samples = j).fit(Customer_Data) 
#   labels = dbscan.labels_ 
#   score.append(silhouette_score(Customer_Data,labels, metric='euclidean'))
#   #print('Silhouetter Score: %.3f' % score + str(i)+str(j))
#  plt.title(str(i)+str(j))
#  plt.plot(range(30,50),score,marker='o') 
#  plt.show()

# dbscan = DBSCAN(eps = 0.005, min_samples = 30).fit(Customer_Data) 
# labels = dbscan.labels_ 
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = labels,cmap='rainbow')
# plt.show() 

#---------------------------------------------------------------

# GMM : 

# gmm = GaussianMixture(n_components=41,covariance_type='spherical').fit(Customer_Data)
# labels = gmm.predict(Customer_Data)
# probs = gmm.predict_proba(Customer_Data)
# print(probs[:3].round(4))
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = labels,cmap='rainbow')
# plt.show() 
    
# gmm = GaussianMixture(n_components=41,covariance_type='tied').fit(Customer_Data)
# labels = gmm.predict(Customer_Data)
# probs = gmm.predict_proba(Customer_Data)
# print(probs[:3].round(4))
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = labels,cmap='rainbow')
# plt.show()   

# gmm = GaussianMixture(n_components=41,covariance_type='diag').fit(Customer_Data)
# labels = gmm.predict(Customer_Data)
# probs = gmm.predict_proba(Customer_Data)
# print(probs[:3].round(4))
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = labels,cmap='rainbow')
# plt.show() 
    
# gmm = GaussianMixture(n_components=41,covariance_type='full').fit(Customer_Data)
# labels = gmm.predict(Customer_Data)
# probs = gmm.predict_proba(Customer_Data)
# print(probs[:3].round(4))
# ax.scatter3D(Customer_Data[:,0], Customer_Data[:,1], Customer_Data[:,2],c = labels,cmap='rainbow')
# plt.show()   