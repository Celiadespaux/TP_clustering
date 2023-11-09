import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################

path = './artificial/'
name="triangle2.arff"

databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

###################### 4.1 (Variation de epsilon et min_pts manuellement) ######################

#for epsilon in [4, 5, 6]:
#    for min_pts in [3, 5, 8, 15]:
#        model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
#        model.fit(datanp)
#        tps2 = time.time()
#        labels = model.labels_

        # Number of clusters in labels, ignoring noise if present.
#        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#        n_noise = list(labels).count(-1)
#        print('Number of clusters: %d' % n_clusters)
#        print('Number of noise points: %d' % n_noise)

#        plt.scatter(f0, f1, c=labels, s=8)
#        plt.title(f"clustering DBSCAN - Epislon= {epsilon} MinPts= {min_pts}\n Number of clusters: {n_clusters} Number of noise points: {n_noise}")
#        plt.show()

###################### 4.2 et 4.3 (Courbe des distances moyennes des k plus proche voisins triées par ordre croissantes) ######################

# Standardisation des donnees
scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)

print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

# Distances aux k plus proches voisins
# Donnees dans X
k = 50
neigh = NearestNeighbors( n_neighbors = k )
neigh.fit(data_scaled)
distances , indices = neigh.kneighbors(data_scaled)
# distance moyenne sur les k plus proches voisins
# en retirant le point " origine "
newDistances = np.asarray ( [ np.average( distances [ i ] [ 1 : ] ) for i in range (0 , distances . shape [ 0 ] ) ] )
# trier par ordre croissant
distancetrie = np.sort( newDistances )
plt.title(" Plus proches voisins " + str ( k ) )
plt.plot(distancetrie)
plt.show()


print("------------------------------------------------------")
print("Appel DBSCAN sur données standardisees ... ")
tps1 = time.time()
epsilon = 0.46
min_pts = k
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(data_scaled)

tps2 = time.time()
labels = model.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
plt.title(f"clustering DBSCAN données standardisées- Epislon= {epsilon} MinPts= {min_pts}\n Number of clusters: {n_clusters} Number of noise points: {n_noise}")
plt.show()

###################### 4.4 HDBSCAN ###################### 

import hdbscan

# Nous avons regrouppé ensemble 3 jeux de données qu'on pense être adaptés pour l'utilisation de DBSCAN (première liste) et 3 non adaptés (deuxième liste)
# Observons si HDBSCAN donne de meilleurs résulats
names = [["banana.arff", "triangle2.arff", "spiral.arff"], ["2d-4c-no4.arff", "2sp2glob.arff", "dense-disk-3000.arff"]]

model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples = 10)

for j in range(2):
    for i, name in enumerate(names[j]):

        databrut = arff.loadarff(open(path+str(name), 'r'))
        datanp = np.array([[x[0],x[1]] for x in databrut[0]])
        scaler = preprocessing.StandardScaler().fit(datanp)
        data_scaled = scaler.transform(datanp)

        cluster_labels = model.fit_predict(data_scaled)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        plt.subplot(2, 3, j*3 + i+1)
        plt.scatter(data_scaled[:,0], data_scaled[:,1], c=cluster_labels, s=8)
        plt.title(f"clustering HDBSCAN données standardisées- \n Number of clusters: {n_clusters} Number of noise points: {n_noise}")
plt.show()

##################################################################