
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Generate sample dataset
X, y = make_blobs(
    n_samples=300,
    centers=4,
    random_state=42
)

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Store results
inertia_values = []
silhouette_scores = []
k_values = range(2, 11)

# 4. Fit K-Means for k = 2 to 10
for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Train model
    labels = model.fit_predict(X_scaled)

    # Calculate inertia
    inertia_values.append(model.inertia_)

    # Calculate silhouette score
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette Score = {score:.4f}"
    )

# 5. Plot Elbow Curve
plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(k_values)
plt.grid(True)

plt.show()

# 6. Plot Silhouette Scores
plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method")
plt.xticks(k_values)
plt.grid(True)

plt.show()

# 7. Find best K using highest silhouette score
best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]

print("\nBest K based on Silhouette Score:", best_k)
