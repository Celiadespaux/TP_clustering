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

path = './artificial/'
name="xclara.arff"

databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
k=3
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

###################### 2.1 ######################

from sklearn.metrics.pairwise import euclidean_distances

data_clusters = [np.array([list(datanp[i]) for i in range(datanp.shape[0]) if labels[i]==j]) for j in range(k)]

dists_points_centers = [euclidean_distances(data_clusters[i], [centroids[i]]) for i in range(k)]

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

### Affichage du resultat du clustering avec visualisation des scores de regroupements ###

plt.figure(figsize=(6, 6))
plt.scatter(datanp[:, 0], datanp[:, 1], c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))

for i in range(len(centroids)):
    circle_max = plt.Circle((centroids[i, 0], centroids[i, 1]), dist_max_clusters[i], color='red', fill=False)
    circle_mean = plt.Circle((centroids[i, 0], centroids[i, 1]), dist_mean_clusters[i], color='black', fill=False, linestyle='dashed')
    plt.gca().add_patch(circle_max)
    plt.gca().add_patch(circle_mean)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie)

###################### 2.2 ######################

inerties = []

for k in range(2,31) : 
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    inerties.append(model.inertia_)

plt.plot(range(2,31),inerties)
plt.xlabel("nb clusters")
plt.ylabel("inertie")
plt.title(f"Inertie en fonction du nombre de cluster ({name})")
plt.show()


###################### 2.3 ######################

from sklearn.metrics import silhouette_score

silhouette_scores = []

time_init = time.time()
for k in range(2,31) : 
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    silhouette_scores.append(silhouette_score(datanp, labels))
    
time_tot = time.time() - time_init 
print(f"Temps total = {np.round(time_tot, 2)}s")

plt.plot(range(2,31),silhouette_scores)
plt.xlabel("nb clusters")
plt.ylabel("silhouette_scores")
plt.title(f"Coefficient de silhouette en fonction du nombre de cluster ({name})")
plt.show()


###################### 2.4 ######################

# Nous avons regrouppé ensemble 3 jeux de données qu'on pense être adaptés pour l'utilisation de k-means (première liste) et 3 non adaptés (deuxième liste)
names = [["DS850.arff", "R15.arff", "spherical_6_2.arff"], ["zelnik1.arff", "banana.arff", "zelnik5.arff"]]

for j in range(2):
    for i, name in enumerate(names[j]):

        databrut = arff.loadarff(open(path+str(name), 'r'))
        datanp = np.array([[x[0],x[1]] for x in databrut[0]])

        silhouette_scores = []
        time_init = time.time()

        for k in range(2,31) : 
            model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
            model.fit(datanp)
            labels = model.labels_
            silhouette_scores.append(silhouette_score(datanp, labels))

        time_tot = np.round(time.time() - time_init, 2)
        plt.subplot(2, 3, i+1)
        plt.scatter(datanp[:, 0], datanp[:, 1], s=8)
        plt.subplot(2, 3, 3 + i+1)
        plt.plot(range(2,31), silhouette_scores)
        plt.title(f"{name} | Total time = {time_tot}")

    plt.show()

###################### 2.5 ######################

names = [["DS850.arff", "R15.arff", "spherical_6_2.arff"], ["zelnik1.arff", "banana.arff", "zelnik5.arff"]]

for j in range(2):
    for i, name in enumerate(names[j]):

        databrut = arff.loadarff(open(path+str(name), 'r'))
        datanp = np.array([[x[0],x[1]] for x in databrut[0]])

        silhouette_scores = []
        time_init = time.time()

        for k in range(2,31) : 
            model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=15, batch_size=8192, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=5, init_size=None, n_init=1, reassignment_ratio=0.01)
            model.fit(datanp)
            labels = model.labels_
            silhouette_scores.append(silhouette_score(datanp, labels))

        time_tot = np.round(time.time() - time_init, 2)
        plt.subplot(2, 3, j*3 + i+1)
        plt.plot(range(2,31), silhouette_scores)
        plt.title(f"{name} | Total time = {time_tot}")

plt.show()





