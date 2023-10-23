"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = '../artificial/'
name="R15.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
#plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
k=15
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances

###################### 2.1

data_clusters = [np.array([list(datanp[i]) for i in range(datanp.shape[0]) if labels[i]==j]) for j in range(k)]

dists_points_centers = [euclidean_distances(data_clusters[i], [centroids[i]]) for i in range(k)] # Not clean because it contains the distances with itself (0)

dist_max_clusters = [max(dists_points_center) for dists_points_center in dists_points_centers]
dist_min_clusters = [min(dists_points_center) for dists_points_center in dists_points_centers]
dist_mean_clusters = [np.mean(dists_points_center) for dists_points_center in dists_points_centers]

print(f"Score de regroupement : \nmax = {dist_max_clusters} \nmin = {dist_min_clusters} \nmean = {dist_mean_clusters}")


dists_centers = euclidean_distances(centroids)
print(dists_centers)

dists_centers = np.array([[v for v in dists_center if v !=0] for dists_center in dists_centers])
print(dists_centers)

dist_max_centers = np.max(dists_centers, axis = 1)
dist_min_centers = np.min(dists_centers, axis = 1)
dist_mean_centers = np.mean(dists_centers, axis = 1)

print(f"Score de séparation : \nmax = {dist_max_centers} \nmin = {dist_min_centers} \nmean = {dist_mean_centers}")



###################### 2.2

inerties = []

for k in range(2,31) : 
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    inerties.append(model.inertia_)

plt.plot(range(2,31),inerties)
plt.xlabel("nb clusters")
plt.ylabel("inertie")
plt.title("inertie en fonction du nombre de cluster")
plt.legend()
plt.show()


###################### 2.3

from sklearn.metrics import silhouette_score

silhouette_scores = []

time_init = time.time()
for k in range(2,31) : 
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    silhouette_scores.append(silhouette_score(datanp, labels))
    
time_tot = time.time() - time_init 
print("temps = " + str(time_tot))

plt.plot(range(2,31),silhouette_scores)
plt.xlabel("nb clusters")
plt.ylabel("silhouette_scores")
plt.title("silhouette_scores en fonction du nombre de cluster")
plt.legend()
plt.show()


###################### 2.4


names = [["2d-4c.arff", "R15.arff", "spherical_6_2.arff"], ["zelnik1.arff", "banana.arff", "DS850.arff"]]

for j in range(2):
    for i, name in enumerate(names[j]):

        databrut = arff.loadarff(open(path+str(name), 'r'))
        datanp = np.array([[x[0],x[1]] for x in databrut[0]])

        silhouette_scores = []

        for k in range(2,31) : 
            model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
            model.fit(datanp)
            labels = model.labels_
            silhouette_scores.append(silhouette_score(datanp, labels))
        plt.subplot(2, 3, j*3 + i+1)
        plt.plot(range(2,31), silhouette_scores)
        plt.title(name)

plt.show()

###################### 2.5

names = [["2d-4c.arff", "R15.arff", "spherical_6_2.arff"], ["zelnik1.arff", "banana.arff", "DS850.arff"]]

for j in range(2):
    for i, name in enumerate(names[j]):

        databrut = arff.loadarff(open(path+str(name), 'r'))
        datanp = np.array([[x[0],x[1]] for x in databrut[0]])

        silhouette_scores = []

        for k in range(2,31) : 
            model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, batch_size=1024, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init='warn', reassignment_ratio=0.01)
            model.fit(datanp)
            labels = model.labels_
            silhouette_scores.append(silhouette_score(datanp, labels))
        plt.subplot(2, 3, j*3 + i+1)
        plt.plot(range(2,31), silhouette_scores)
        plt.title(name)

plt.show()





