import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        self.dataset_file = dataset_file
        self.data = None
        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        self.data = mat['X']
        print(f"Data loaded. Shape of 'X': {self.data.shape}")

    def model_fit(self, n_clusters=3):
        '''
        Initialize self.model and execute KMeans clustering on self.data
        '''
        self.model = KMeans(n_clusters=n_clusters, random_state=0)
        self.model.fit(self.data)
        return self.model.cluster_centers_, self.model.labels_

    def plot_clusters(self, cluster_centers, labels):
        '''
        Plot the data points and cluster centers
        '''
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Centroids')
        plt.title('KMeans Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default="dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    cluster_centers, labels = classifier.model_fit(n_clusters=7)
    print("Cluster Centers:")
    print(cluster_centers)
    
    classifier.plot_clusters(cluster_centers, labels)