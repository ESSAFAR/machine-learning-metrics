import numpy as np
import pandas as pd

from utils.dataset import generate_clustering_test_dataset


def pairwise_distances(data):
    return np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))


def silhouette_score(data, labels):
    distances = pairwise_distances(data)
    unique_labels = np.unique(labels)
    silhouettes = []

    for i in range(len(data)):
        same_cluster = labels == labels[i]
        other_clusters = labels != labels[i]

        a_i = distances[i, same_cluster].mean()
        b_i = np.min(
            [distances[i, labels == lbl].mean() for lbl in unique_labels if lbl != labels[i]])
        silhouettes.append((b_i - a_i) / max(a_i, b_i))

    return np.mean(silhouettes)


def calinski_harabasz_index(data, labels):
    overall_mean = np.mean(data, axis=0)
    unique_labels = np.unique(labels)
    n_samples = len(data)
    n_clusters = len(unique_labels)

    between_cluster_sum = 0
    within_cluster_sum = 0

    for lbl in unique_labels:
        cluster_points = data[labels == lbl]
        cluster_mean = np.mean(cluster_points, axis=0)
        n_points = len(cluster_points)
        between_cluster_sum += n_points * np.sum((cluster_mean - overall_mean) ** 2)
        within_cluster_sum += np.sum((cluster_points - cluster_mean) ** 2)

    return (between_cluster_sum / (n_clusters - 1)) / (within_cluster_sum / (n_samples - n_clusters))


def davies_bouldin_index(data, labels):
    unique_labels = np.unique(labels)
    cluster_means = [np.mean(data[labels == lbl], axis=0) for lbl in unique_labels]
    cluster_variances = [np.mean(np.sqrt(np.sum((data[labels == lbl] - cluster_means[i]) ** 2, axis=1)))
                         for i, lbl in enumerate(unique_labels)]

    db_scores = []
    for i in range(len(unique_labels)):
        max_ratio = 0
        for j in range(len(unique_labels)):
            if i != j:
                inter_cluster_distance = np.linalg.norm(cluster_means[i] - cluster_means[j])
                ratio = (cluster_variances[i] + cluster_variances[j]) / inter_cluster_distance
                max_ratio = max(max_ratio, ratio)
        db_scores.append(max_ratio)

    return np.mean(db_scores)


if __name__ == "__main__":
    df = generate_clustering_test_dataset()
    features = df.drop(columns=['Cluster_Labels']).values
    labels = df['Cluster_Labels'].values

    silhouette = silhouette_score(features, labels)
    calinski_harabasz = calinski_harabasz_index(features, labels)
    davies_bouldin = davies_bouldin_index(features, labels)

    print("Silhouette Score:", silhouette)
    print("Calinski-Harabasz Index:", calinski_harabasz)
    print("Davies-Bouldin Index:", davies_bouldin)
