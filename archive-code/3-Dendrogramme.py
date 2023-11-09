import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram

path = './artificial/'
name="square1.arff"

databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne


from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix) #, **kwargs)


###################### 3.1 ######################

linkages = ["ward", "complete", "average", "single"]

for linkage_type in linkages:
    # setting distance_threshold=0 ensures we compute the full tree.
    model = cluster.AgglomerativeClustering(distance_threshold=0, linkage=linkage_type, n_clusters=None)

    model = model.fit(datanp)
    plt.figure(figsize=(12, 12))
    plt.title(f"Hierarchical Clustering Dendrogram with linkage '{linkage_type}'")
    # plot the top p levels of the dendrogram
    plot_dendrogram(model) #, truncate_mode="level", p=5)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

###################### 3.2 (Temps de calculs) ######################

###
# FIXER la distance
###
time_linkages = []

for linkage_type in linkages:
    time_init = time.time()
    for seuil_dist in range(1, 20):
        model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage=linkage_type, n_clusters=None)
        model = model.fit(datanp)
    time_linkages.append(np.round(time.time() - time_init, 2))
  
print(f"Temps de calcul en faisant varier le seuil de distance pour les différents types de linkages : \n{time_linkages}")

###
# FIXER le nombre de clusters
###
time_linkages = []

for linkage_type in linkages:
    time_init = time.time()
    for k in range(2, 30):
        model = cluster.AgglomerativeClustering(linkage=linkage_type, n_clusters=k)
        model = model.fit(datanp)
    time_linkages.append(np.round(time.time() - time_init, 2))

print(f"Temps de calcul en faisant varier le nombre de clusters pour les différents types de linkages : \n{time_linkages}")

#######################################################################