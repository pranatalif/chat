import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.neighbors.nearest_centroid import NearestCentroid
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances

# chaos = pd.read_csv('dataset/TestSDN.csv')
chaos = pd.read_csv(
    '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/dataset/ct_baseline.csv')
# chaos = pd.read_csv(
# '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/Second Paper/Results/Scenario 1/V2/TestSDN Data Set - Sent.csv')
# /Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/dataset
# /Users/aalif/Documents/Google Drive/4.Prancis/Paper/Second Paper/Results/Scenario 1/V3/New TestSDN Data Set - S1, E1
chaos.head()

# Select Columns (Variables) on which to run PCA
X = chaos.loc[:, 'num_err_msg':'uptime'].values
# X = chaos.loc[:, 'Execution 1':'Execution 1'].values
y = chaos.loc[:, 'status'].values
data = chaos.loc[:, 'data'].values

# If we do not specify how many components, all are included
pca = PCA()
X_r = pca.fit(X).transform(X)

X_std = StandardScaler().fit_transform(X)
X_r = pca.fit(X_std).transform(X_std)

target_names = chaos.iloc[:, 10].unique()
x = chaos.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].values

print("target_names : ", target_names)

# kmeans = KMeans(n_clusters=5)
# labels = kmeans.fit_predict(x)
# y_kmeans = kmeans.predict(X_r)
# print("kmeans : ", kmeans)
# print("labels labels: ", labels.labels_)
# print("labels labels: ", labels)


# def kmeanss():
#     plt.figure()
#     plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis')

#     centers = kmeans.cluster_centers_
#     print("kmeans : ", kmeans)
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=None)

#     plt.show()


def pca_scatter(pca1, pca2, data):
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 20})
    plt.close
    plt.figure(figsize=(10, 7))
    colors = ['lime', 'blue', 'chocolate', 'purple', 'magenta',
              'maroon', 'crimson', 'violet', 'yellow', 'yellowgreen',
              'khaki', 'indigo', 'sienna', 'plum', 'hotpink',
              'saddlebrown', 'coral', 'gray', 'orchid', 'k',
              'tomato', 'aqua', 'turquoise', 'crimson', 'salmon']
    # markers = ['$a$', 's', 'x', 'o', 'v', '>',
    #            '<', '^', 'h', '+', '*',
    #            'p', 'P', 'D', 'd', 'X',
    #            '1', '2', '2', '4', '8',
    #            '|', '_', 'H', '.']

    # print(type(colors))

   # for color, target_name in zip(colors, target_names):
   #     plt.scatter(X_r[y == target_name, pca1], X_r[y == target_name, pca2], color=color, alpha=None, lw=lw,
   #                 label=target_name)
   # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('PCA: Components {} and {}'.format(pca1, pca2))
    # plt.title('Scenario 1: One video server crashes')
   # plt.xlabel('Component {}'.format(pca1))
   # plt.ylabel('Component {}'.format(pca2))
   # plt.show()

    # for color, target_name, marker in zip(colors, target_names, markers):
    #     plt.scatter(X_r[y == target_name, pca1], X_r[y ==
    #
    #                                                target_name, pca2], color=color, label=target_name, marker=markers)
    arr_data = np.empty((0, 2), int)
    arr_centroid = np.empty((0, 2), int)

    for color, target_name in zip(colors, target_names):
        print("target_name : ", target_name)
        x_pca1 = np.array(X_r[y == target_name, pca1])
        x_pca2 = np.array(X_r[y == target_name, pca2])
        arr = np.stack((x_pca1, x_pca2), axis=1)
        plt.scatter(X_r[y == target_name, pca1], X_r[y ==
                                                     target_name, pca2], s=100, color=color, label=target_name)
        print("x_pca1 = ", x_pca1)
        print("x_pca2 = ", x_pca2)
        print("arr = ", arr)
        # plt.annotate(data, (colors, target_names))
        kmeans = KMeans(n_clusters=1)
        kfit = kmeans.fit(arr)
        centroid = kfit.cluster_centers_
        # centroid = np.append(arr1, centroid, axis=0)
        # plt.scatter(kmeans.cluster_centers_[ // cluster centroid
        #             :, 0], kmeans.cluster_centers_[:, 1], color='black', marker="*")

        # print("data point = ", arr)
        # print("centroid = ", kfit.cluster_centers_)

        # dist = np.linalg.norm(arr-centroid)
        # print(dist)
        arr_data = np.append(arr_data, arr, axis=0)
        arr_centroid = np.append(arr_centroid, kfit.cluster_centers_, axis=0)

        i = 0
        for centroid in arr_centroid:
            for data in arr_data:
                if i < np.size(arr_data, 0):
                    dist = np.linalg.norm(arr_data[i]-arr_centroid, axis=1)
                    print(i+1, "distance:", dist, "min_dist:", np.argmin(dist))
                    i = i+1
                else:
                    break
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('PCA: Components {} and {}'.format(pca1+1, pca2+1))
    plt.xlabel('Principal component {}'.format(pca1+1))
    plt.ylabel('Principal component {}'.format(pca2+1))
    # for m, n in zip(X_r[y == target_name, pca1], X_r[y ==
    #                                                  target_name, pca2]):
    #     label = "{:.1f}".format(n)
    #     plt.annotate(data,  # this is the text
    #                  (m, n),  # this is the point to label
    #                  textcoords="offset points",  # how to position the text
    #                  xytext=(0, 8),  # distance from text to points (x,y)
    #                  ha='center')  # horizontal alignment can be left, right or center
    plt.show()


pca_scatter(0, 1, data)

# kmeanss()

# source: https://jmausolf.github.io/code/pca_in_python/; k-means: https://heartbeat.fritz.ai/k-means-clustering-using-sklearn-and-python-4a054d67b187
# We also explore the possibility of defining a distance score between the nominal and the observed behavior by the PCA. We plan to evaluate whether this distance score is correlated with the identification of a configuration error.

# arr.
# arr_data.clear
# arr_centroid.clear


# print("A arr_Data:", arr_data)
# print("A arr_Centroid:", arr_centroid)
# print("range ", range(arr_data))

# Alternative partial vectorized solution.
# Iterate over training examples.
# for i in range(X.shape[0]):
#     distances = np.linalg.norm(X[i] - centroids, axis=1)
# argmin returns the indices of the minimum values along an axis,
# replacing the need for a for-loop and if statement.
# min_dst = np.argmin(distances)
# idx[i] = min_dst
# distance.euclidean([1, 0, 0], [0, 1, 0])

# print(i+1, "distance:", dist,
#       "min_dist:", np.argmin(dist))

# for centroid in arr_centroid:
#     for data in arr_data:
#         dist = np.linalg.norm(arr_data[-3]-arr_centroid, axis=1)
# print("distance: ", dist)
