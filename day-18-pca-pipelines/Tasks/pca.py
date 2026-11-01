
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = Path(__file__).parent / "Mall_Customers.csv"
df = pd.read_csv(DATA_PATH)


print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# 2. SELECT FEATURES
# ==========================================

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

print("\nSelected Features Shape:", X.shape)

# ==========================================
# 3. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# 4. APPLY K-MEANS CLUSTERING
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
df["Cluster"] = clusters

print("\nCluster Distribution:")
print(df["Cluster"].value_counts().sort_index())

# ==========================================
# 5. APPLY PCA (2 COMPONENTS)
# ==========================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nOriginal Data Shape:", X_scaled.shape)
print("PCA Reduced Data Shape:", X_pca.shape)

# ==========================================
# 6. REPORT EXPLAINED VARIANCE
# ==========================================

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nExplained Variance Percentage:")
print(pca.explained_variance_ratio_ * 100)

total_variance = pca.explained_variance_ratio_.sum()

print("\nTotal Explained Variance:")
print(f"{total_variance * 100:.2f}%")

# ==========================================
# 7. PLOT PCA CLUSTERS
# ==========================================

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Customer Clusters Using PCA")

plt.colorbar(label="Cluster")

plt.grid(True)
plt.show()
