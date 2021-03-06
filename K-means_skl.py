import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans

style.use('ggplot')

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

model = KMeans(n_clusters=2)
model.fit(X)

centroids = model.cluster_centers_
labels = model.labels_

colors = 10*['g.','r.','c.','b.','k.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)

plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
plt.show()