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
import scipy.cluster.hierarchy as shc
from yellowbrick.cluster import SilhouetteVisualizer



customer = pd.read_csv("/Users/wegdan/Desktop/Semester 10 🥳/Machine Learning/Assignment 1/Customer data.csv")
Customer_Data=np.vstack((customer.Sex,customer.Maritalstatus,customer.Age,customer.Education,customer.Income,customer.Occupation,customer.Settlementsize)).T

#KMeans : 

# Elbow Method

# inertias = []
# for i in range(1,21):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(list(Customer_Data))
#     inertias.append(kmeans.inertia_)

# plt.plot(range(1,21), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# #plt.show()

# #---------------------------------------------------------------

# # Clustring

# z=2
# for n in range(4,11,3):
 
#  kmeans = KMeans(n_clusters=n,random_state = 0)
#  y_kmeans=kmeans.fit_predict(Customer_Data)
#  visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
#  visualizer.fit(Customer_Data)
#  #visualizer.show()
#  score = silhouette_score(Customer_Data, kmeans.labels_, metric='euclidean')
#  print('Silhouetter Score: %.3f' % score + " no of clusters= " + str(n))

#  for k in range(0,7):
  
#   for l in range(1,7):
   
#    if(l>k):
#     plt.figure(z)
#     for j in range(0,n):
#      plt.title("feature  " + str(k)+ " with  "+ str(l) + "  no. of clusters= "+ str(n))
#      plt.scatter(Customer_Data[y_kmeans==j,k], Customer_Data[y_kmeans== j,l])
#      plt.scatter(kmeans.cluster_centers_[:, k], kmeans.cluster_centers_[:,l], label = 'Centroids')
#      #plt.savefig('/Users/wegdan/Desktop/Semester 10 🥳/Machine Learning/Assignment 1/figs/fig)'+str(z))
#      z+=1
  
#plt.show()

# #---------------------------------------------------------------

#Hierarchal : 


# average 

# for n in range(4,11,3):
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

# #  visualizer = SilhouetteVisualizer(ac1, colors='yellowbrick')
# #  visualizer.fit(Customer_Data)
# #  visualizer.show()

#  score = silhouette_score(Customer_Data, ac1.labels_ , metric='cosine')
#  print('Silhouetter Score: %.3f' % score + " cosine average no of clusters= " + str(n))
#  score = silhouette_score(Customer_Data, ac4.labels_, metric='cosine')
#  print('Silhouetter Score: %.3f' % score + " cosine single  no of clusters= " + str(n))

#  score = silhouette_score(Customer_Data, ac2.labels_, metric='euclidean')
#  print('Silhouetter Score: %.3f' % score + " euclidean average no of clusters= " + str(n))
#  score = silhouette_score(Customer_Data, ac5.labels_, metric='euclidean')
#  print('Silhouetter Score: %.3f' % score + " euclidean single  no of clusters= " + str(n))

#  score = silhouette_score(Customer_Data, ac3.labels_, metric='manhattan')
#  print('Silhouetter Score: %.3f' % score + " manhattan average no of clusters= " + str(n))
#  score = silhouette_score(Customer_Data, ac6.labels_, metric='manhattan')
#  print('Silhouetter Score: %.3f' % score + " manhattan single  no of clusters= " + str(n))


#  for k in range(0,7):
  
#   for l in range(1,7):
   
#    if(l>k):
#     plt.figure()
#     plt.title("feature  " + str(k)+ " with  "+ str(l) + "  no. of clusters= "+ str(n) + "  aff=cosine link=average ")
#     plt.scatter(Customer_Data[:,k], Customer_Data[:,l],c = ac1.fit_predict(Customer_Data), cmap ='rainbow')

#  for k in range(0,7):
  
#   for l in range(1,7):
   
#    if(l>k):
#     plt.figure()
#     plt.title("feature  " + str(k)+ " with  "+ str(l) + "  no. of clusters= "+ str(n) + "  aff=euclidean link=average ")
#     plt.scatter(Customer_Data[:,k], Customer_Data[:,l],c = ac2.fit_predict(Customer_Data), cmap ='rainbow')

#  for k in range(0,7):
  
#   for l in range(1,7):
   
#    if(l>k):
#     plt.figure()
#     plt.title("feature  " + str(k)+ " with  "+ str(l) + "  no. of clusters= "+ str(n) + "  aff=manhattan link=average ")
#     plt.scatter(Customer_Data[:,k], Customer_Data[:,l],c = ac3.fit_predict(Customer_Data), cmap ='rainbow')

#  for k in range(0,7):
  
#   for l in range(1,7):
   
#    if(l>k):
#     plt.figure()
#     plt.title("feature  " + str(k)+ " with  "+ str(l) + "  no. of clusters= "+ str(n) + "  aff=cosine link=single ")
#     plt.scatter(Customer_Data[:,k], Customer_Data[:,l],c = ac4.fit_predict(Customer_Data), cmap ='rainbow')

#  for k in range(0,7):
  
#   for l in range(1,7):
   
#    if(l>k):
#     plt.figure()
#     plt.title("feature  " + str(k)+ " with  "+ str(l) + "  no. of clusters= "+ str(n) + "  aff=euclidean link=single ")
#     plt.scatter(Customer_Data[:,k], Customer_Data[:,l],c = ac5.fit_predict(Customer_Data), cmap ='rainbow')

#  for k in range(0,7):
  
#   for l in range(1,7):
   
#    if(l>k):
#     plt.figure()
#     plt.title("feature  " + str(k)+ " with  "+ str(l) + "  no. of clusters= "+ str(n) + "  aff=manhattan link=single ")
#     plt.scatter(Customer_Data[:,k], Customer_Data[:,l],c = ac6.fit_predict(Customer_Data), cmap ='rainbow')


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



# # #---------------------------------------------------------------

# DBScan : 
# np.set_printoptions(threshold=sys.maxsize)
# for i in np.arange (0.1,3,0.4):
#  for j in range (5,30,3):
#   dbscan = DBSCAN(eps = 1000, min_samples = j).fit(Customer_Data) 
#   labels = dbscan.labels_ 
#   score = silhouette_score(Customer_Data,labels, metric='euclidean')
#   print('Silhouetter Score: %.3f' % score + str(i))
#   for k in range(0,7):
  
#    for l in range(1,7):
   
#     if(l>k):
#      plt.scatter(Customer_Data[:, k], Customer_Data[:,l], c = labels, cmap= "rainbow") 
    
#      plt.show() 
 #---------------------------------------------------------------

# GMM : 

for k in range(0,7):
  for l in range(1,7):
   if(l>k):
     gmm = GaussianMixture(n_components=5,covariance_type='spherical').fit(Customer_Data)
     labels = gmm.predict(Customer_Data)
     plt.scatter(Customer_Data[:, k], Customer_Data[:, l], c=labels, s=40, cmap='viridis')
    
     gmm = GaussianMixture(n_components=5,covariance_type='tied').fit(Customer_Data)
     labels = gmm.predict(Customer_Data)
     plt.scatter(Customer_Data[:, k], Customer_Data[:, l], c=labels, s=40, cmap='viridis')
     plt.show() 

     gmm = GaussianMixture(n_components=5,covariance_type='diag').fit(Customer_Data)
     labels = gmm.predict(Customer_Data)
     plt.scatter(Customer_Data[:, k], Customer_Data[:, l], c=labels, s=40, cmap='viridis')
     plt.show() 

     gmm = GaussianMixture(n_components=5,covariance_type='full').fit(Customer_Data)
     labels = gmm.predict(Customer_Data)
     plt.scatter(Customer_Data[:, k], Customer_Data[:, l], c=labels, s=40, cmap='viridis')
     plt.show() 