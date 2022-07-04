import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import seaborn as sns
import multiprocessing as mp


def mp_kmeans(X, k):
    diff = True
    # initialize clusters array
    clusters = np.zeros(X.shape[0])
    # choose random initial centroids
    centroids = X[np.random.randint(X.shape[0], size=k), :]

    # Pool
    pool = mp.Pool(mp.cpu_count())

    while diff:

        distances = pool.map(get_distance, [centroid for centroid in centroids])
        clusters = np.argmin(np.array(distances).transpose(), axis=1)
        new_centroids = pd.DataFrame(X).groupby(by=clusters).mean().values

        # if centroids are same then leave
        if np.count_nonzero(centroids - new_centroids) == 0:
            diff = False
        else:
            centroids = new_centroids

    pool.close()
    return centroids, clusters


def get_distance(centroid):
    return np.sqrt(np.square(X - centroid).sum(axis=1))


# Read data
computers = pd.read_csv('computers.csv', index_col='id')
computers['cd'] = [0 if x == 'no' else 1 for x in computers['cd']]
computers['laptop'] = [0 if x == 'no' else 1 for x in computers['laptop']]
X = computers.to_numpy()


if __name__ == '__main__':

    print(f'Dimensions data: {X.shape}')

    # from the plot we saw that the optimal number of clusters is 2
    optimal_k = 2

    # Start time
    start_time = time.time()

    # Predicting the clusters
    centroids, clusters = mp_kmeans(X, optimal_k)

    # Final time
    print("--- %s seconds ---" % (time.time() - start_time))

    # Cluster with highest average price
    print(f"The cluster with highest average price has a value of: "
          f"{max(np.mean(computers[clusters == 0]['price']), np.mean(computers[clusters == 1]['price']))}")

    # Variables to print
    # 'price', 'speed', 'hd', 'ram', 'screen', 'cores', 'cd', 'laptop', 'trend'
    pos_x = 0
    pos_y = 1
    var1 = computers.columns[pos_x]  # x_axis
    var2 = computers.columns[pos_y]  # y_axis

    # Scatter plot
    computers_0 = computers[clusters == 0]
    computers_1 = computers[clusters == 1]
    plt.scatter(computers_0.loc[:, var1].values, computers_0.loc[:, var2].values, color='red', label='Cluster 1')
    plt.scatter(computers_1.loc[:, var1].values, computers_1.loc[:, var2].values, color='blue', label='Cluster 2')
    plt.scatter(centroids[:, pos_x], centroids[:, pos_y], s=80, color='black')
    plt.legend(loc='lower right')
    plt.title('Scatter plot')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.show()

    # Normalization
    norm_centroids = np.zeros(centroids.shape)
    for idx, feature in enumerate(computers.columns):
        norm_centroids[0, idx] = (centroids[0, idx] - np.mean(computers[feature])) / np.std(computers[feature])
        norm_centroids[1, idx] = (centroids[1, idx] - np.mean(computers[feature])) / np.std(computers[feature])

    # Heatmap
    df_heat = pd.DataFrame(norm_centroids.transpose(), index=computers.columns.tolist(), columns=range(1, optimal_k+1))
    sns.heatmap(df_heat)
    plt.title('Heatmap')
    plt.ylabel('Attributes')
    plt.xlabel('Cluster')
    plt.show()
