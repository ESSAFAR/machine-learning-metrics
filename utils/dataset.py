import numpy as np
import pandas as pd


def generate_classification_test_dataset(random_state=42):
    np.random.seed(random_state)
    data = np.random.rand(100, 5)
    labels = np.random.randint(0, 2, size=100)
    probabilities = np.random.rand(100)
    predictions = (probabilities >= 0.5).astype(int)

    dataset = pd.DataFrame(data, columns=[f"Feature_{i + 1}" for i in range(data.shape[1])])
    dataset['True_Label'] = labels
    dataset['Predicted_Label'] = predictions
    dataset['Predicted_Probs'] = probabilities
    return dataset


def generate_regression_test_dataset(random_state=42):
    np.random.seed(random_state)
    n_samples = 100
    true_values = np.random.uniform(0, 100, size=n_samples)
    noise = np.random.normal(0, 10, size=n_samples)
    predicted_values = true_values + noise
    return pd.DataFrame({'True_Values': true_values, 'Predicted_Values': predicted_values})


def generate_clustering_test_dataset(n_samples=500, n_features=2, n_clusters=4, random_state=42):
    np.random.seed(random_state)
    centers = np.random.uniform(-10, 10, (n_clusters, n_features))
    data = []
    labels = []
    for i, center in enumerate(centers):
        cluster_data = center + np.random.randn(n_samples // n_clusters, n_features)
        data.append(cluster_data)
        labels.extend([i] * (n_samples // n_clusters))
    data = np.vstack(data)
    df = pd.DataFrame(data, columns=[f"Feature_{i + 1}" for i in range(n_features)])
    df['Cluster_Labels'] = labels
    return df