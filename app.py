import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Streamlit App

st.title("Customer Segmentation")

# Upload dataset
uploaded_file = st.file_uploader(r"C:\Users\Bhagvan\Customer Segment\marketing_campaign.xlsx")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
else:
    st.stop()    

# Data Preprocessing
st.subheader("Data Preprocessing")

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
st.write("Numeric Features used for clustering:")
st.write(list(numeric_df.columns))

# Handle missing values
numeric_df = numeric_df.dropna()   

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# KMeans Clustering
st.subheader("KMeans Clustering")
k = st.slider("Select number of clusters (K)", 2, 10, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

numeric_df['KMeans_Cluster'] = kmeans_labels
st.write("KMeans Clustering Results")
st.dataframe(numeric_df.head())

# Silhouette Score
kmeans_score = silhouette_score(X_scaled, kmeans_labels)
st.write(f"Silhouette Score (KMeans): {kmeans_score:.3f}")

# Plot clusters (first 2 features)
fig, ax = plt.subplots()
scatter = ax.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans_labels, cmap='viridis')
ax.set_title("KMeans Clusters")
st.pyplot(fig)

# Hierarchical Clustering
st.subheader("Hierarchical Clustering (Agglomerative)")

h_clusters = st.slider("Select number of clusters (Hierarchical)", 2, 10, 3)
agg = AgglomerativeClustering(n_clusters=h_clusters)
agg_labels = agg.fit_predict(X_scaled)

numeric_df['Hierarchical_Cluster'] = agg_labels
st.write("Hierarchical Clustering Results")
st.dataframe(numeric_df.head())

# Silhouette Score
hier_score = silhouette_score(X_scaled, agg_labels)
st.write(f"Silhouette Score (Hierarchical): {hier_score:.3f}")

# Dendrogram
st.subheader("Dendrogram")
Z = linkage(X_scaled, method='ward')
fig, ax = plt.subplots(figsize=(8, 4))
dendrogram(Z, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
st.pyplot(fig)

# DBSCAN
st.subheader("DBSCAN Clustering")

eps_val = st.slider("Epsilon (eps)", 0.1, 10.0, 1.0, 0.1)
min_samples_val = st.slider("Minimum Samples", 2, 20, 5)
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
dbscan_labels = dbscan.fit_predict(X_scaled)

numeric_df['DBSCAN_Cluster'] = dbscan_labels
st.write("DBSCAN Clustering Results")
st.dataframe(numeric_df.head())

# Check if more than 1 cluster for silhouette score
if len(set(dbscan_labels)) > 1:
    db_score = silhouette_score(X_scaled, dbscan_labels)
    st.write(f"Silhouette Score (DBSCAN): {db_score:.3f}")
else:
    st.write("DBSCAN formed only one cluster. Silhouette score not available.")

# Plot DBSCAN results
fig, ax = plt.subplots()
scatter = ax.scatter(X_scaled[:,0], X_scaled[:,1], c=dbscan_labels, cmap='plasma')
ax.set_title("DBSCAN Clusters")
st.pyplot(fig)

# Final Output
st.subheader("Final Dataset with Clusters")
st.dataframe(numeric_df.head(20))
