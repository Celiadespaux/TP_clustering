import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = '../artificial/'
name="square1.arff"

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
plt.show()



### FIXER la distance
# 


from sklearn.metrics import silhouette_score

silhouette_scores = []

for seuil_dist in range(1,15) : 
    model = cluster.AgglomerativeClustering(linkage='average', distance_threshold=seuil_dist, n_clusters=None)
    model = model.fit(datanp)
    labels = model.labels_

    print(set(labels))
    if len(set(labels))==1:
        silhouette_scores.append(0)
    else:
        silhouette_scores.append(silhouette_score(datanp, labels))
    

plt.plot(range(1,15),silhouette_scores)
plt.xlabel("seuil de distance")
plt.ylabel("silhouette_scores")
plt.title("silhouette_scores en fonction du seuil de distance avec le linkage average")
plt.legend()
plt.show()


names = [["2d-4c.arff", "R15.arff", "3MC.arff"], ["2sp2glob.arff", "3-spiral.arff", "rings.arff"]]

for j in range(2):
    for i, name in enumerate(names[j]):

        databrut = arff.loadarff(open(path+str(name), 'r'))
        datanp = np.array([[x[0],x[1]] for x in databrut[0]])

        silhouette_scores = []

        for k in range(2,31) : 
            model = cluster.AgglomerativeClustering(linkage='ward', n_clusters=k)
            model = model.fit(datanp)
            labels = model.labels_
            if len(set(labels))==1:
                silhouette_scores.append(0)
            else:
                silhouette_scores.append(silhouette_score(datanp, labels))
        plt.subplot(2, 3, j*3 + i+1)
        plt.plot(range(2,31), silhouette_scores)
        plt.title(name)

plt.show()




###
# FIXER le nombre de clusters
###

silhouette_scores = []

for k in range(2,31) : 
    model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
    model = model.fit(datanp)
    labels = model.labels_
    silhouette_scores.append(silhouette_score(datanp, labels))
    

plt.plot(range(2,31),silhouette_scores)
plt.xlabel("nb clusters")
plt.ylabel("silhouette_scores")
plt.title("silhouette_scores en fonction du nombre de cluster avec le linkage average")
plt.legend()
plt.show()







#######################################################################