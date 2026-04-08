import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Read dataset from CSV file
data = pd.read_csv("data.csv")

# Convert dataset to array
X = data.values

# Number of clusters
k = 3

# ---------------------------
# K-Means Clustering
# ---------------------------
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans_labels = kmeans.fit_predict(X)

# Calculate silhouette score
kmeans_score = silhouette_score(X, kmeans_labels)

# ---------------------------
# EM Algorithm (GMM)
# ---------------------------
gmm = GaussianMixture(n_components=k, random_state=0)
gmm_labels = gmm.fit_predict(X)

# Calculate silhouette score
gmm_score = silhouette_score(X, gmm_labels)

# Display results
print("K-Means Silhouette Score:", kmeans_score)
print("EM Silhouette Score:", gmm_score)
