import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import time
import seaborn as sns


computers = pd.read_csv('computers.csv', index_col='id')

computers['cd'] = [0 if x == 'no' else 1 for x in computers['cd']]
computers['laptop'] = [0 if x == 'no' else 1 for x in computers['laptop']]

X = computers.to_numpy()

'''# Check optimal number of clusters
n_clusters = range(1, 6)
kmeans = [KMeans(n_clusters=i) for i in n_clusters]
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
# Elbow graph
plt.plot(n_clusters,score)
plt.xlabel('Number of Clusters')
plt.xticks(n_clusters)
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()'''

# from the plot we see that the optimal number of clusters is 2
optimal_k = 2

# Start time
start_time = time.time()

# Training
kmeans = KMeans(n_clusters=optimal_k).fit(X)
# Predicting the clusters
labels = kmeans.predict(X)

# Final time
print("--- %s seconds ---" % (time.time() - start_time))

# Getting the cluster centers
centroids = kmeans.cluster_centers_

# Cluster with highest average price
print(f"The cluster with highest average price has a value of: "
      f"{max(np.mean(computers[labels == 0]['price']), np.mean(computers[labels == 1]['price']))}")

# Variables to print
var1 = 'price' # x_axis
var2 = 'speed' # y_axis

# Scatter plot
computers_0 = computers[labels == 0]
computers_1 = computers[labels == 1]
plt.scatter(computers_0.loc[:, var1].values, computers_0.loc[:, var2].values, color='red')
plt.scatter(computers_1.loc[:, var1].values, computers_1.loc[:, var2].values, color='blue')
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='black')
plt.xlabel(var1)
plt.ylabel(var2)
plt.show()

# Normalization
norm_centroids = np.zeros(centroids.shape)
for idx, feature in enumerate(computers.columns):
    norm_centroids[0, idx] = (centroids[0, idx] - np.mean(computers[feature]))/np.std(computers[feature])
    norm_centroids[1, idx] = (centroids[1, idx] - np.mean(computers[feature]))/np.std(computers[feature])

# Heatmap
index_heat = computers.columns.tolist()
df_heat = pd.DataFrame(norm_centroids.transpose(), index=index_heat, columns=range(1, optimal_k+1))
sns.heatmap(df_heat)
plt.title('Heatmap')
plt.ylabel('Attributes')
plt.xlabel('Cluster')
plt.show()




