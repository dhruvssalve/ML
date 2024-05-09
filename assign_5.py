from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the datasetj
url="https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/8bd6144a87988213693754baaa13fb204933282d/Mall_Customers.csv"
data = pd.read_csv("Mall_Customers.csv")
X = data.iloc[:, [2, 3]].values
# Preprocess the data if necessary (scaling, normalization, etc.)

# Determine the optimal number of clusters using silhouette method
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores for different number of clusters
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method for Optimal K')
plt.show()

# Choose the optimal number of clusters with the highest silhouette score
optimal_k = np.argmax(silhouette_scores) + 2  # Add 2 because range starts from 2
print("Optimal number of clusters:", optimal_k)

# Perform K-means clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Calculate silhouette score for each data point
silhouette_values = silhouette_samples(X, cluster_labels)

# Print the average silhouette score
print("Average silhouette score:", silhouette_score(X, cluster_labels))

# Visualize the silhouette scores
y_lower = 10
plt.figure(figsize=(8, 6))
for i in range(optimal_k):
    cluster_silhouette_values = silhouette_values[cluster_labels == i]
    cluster_silhouette_values.sort()
    cluster_size = cluster_silhouette_values.shape[0]
    y_upper = y_lower + cluster_size

    color = plt.cm.nipy_spectral(float(i) / optimal_k)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
    y_lower = y_upper + 10

plt.title("Silhouette plot for KMeans clustering")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
plt.axvline(x=silhouette_score(X, cluster_labels), color="red", linestyle="--")
plt.yticks([])  # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()
