from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X=np.array([[1,2,3,5],
[2,1,1,3],
[1,5,2,2],
[2,2,3,1],
[2,5,4,1],
[4,4,1,4],
[4,3,1,4],
[3,3,4,2]])

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

print("data\t\tclasse")

for i in range(0,8):
    print(X[i],"\t",y_km[i])


x_test = np.array([[1.25, 2.76, 4.22, 1.07]])
#x_test=x_test.reshape(-1,1)
print("\n\nLa classe de cette entrée est : ",km.predict(x_test))



# plot the 3 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()