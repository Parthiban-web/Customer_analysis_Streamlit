import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Streamlit page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧩 Customer Segmentation Using K-Means Clustering")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your customer dataset (Excel)", type=["xlsx"])
if uploaded_file is not None:
    # Load dataset
    C = pd.read_excel(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(C.head())

    # Data Preprocessing
    drop_cols = ['invoice_no', 'customer_id', 'invoice_date']
    for col in drop_cols:
        if col in C.columns:
            C = C.drop(col, axis=1)
    
    # Encode gender
    if 'gender' in C.columns:
        C['gender'] = C['gender'].map({'Male': 0, 'Female': 1})
    
    # One-hot encode categorical columns
    cat_cols = ['category', 'payment_method', 'shopping_mall']
    for col in cat_cols:
        if col in C.columns:
            C = pd.get_dummies(C, columns=[col], drop_first=True)

    st.subheader("Preprocessed Data")
    st.dataframe(C.head())

    # Feature Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(C)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    st.subheader("PCA Visualization of Customer Data")
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(pca_data[:,0], pca_data[:,1], s=50)
    ax.set_title("PCA of Customers")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    # Find the best K using silhouette score
    st.subheader("Finding Optimal Number of Clusters")
    silhouette_scores = []
    K = range(2, 26)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, clusters)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(K, silhouette_scores, 'bo-')
    ax2.set_title("Silhouette Scores for Different K Values")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True)
    st.pyplot(fig2)

    # Best K
    best_k = K[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    st.success(f"✅ Best K: {best_k} with Silhouette Score: {best_score:.3f}")

    # Final KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    C['Cluster'] = kmeans.fit_predict(scaled_data)

    # Visualize Final Clusters
    st.subheader("Final Cluster Visualization (PCA)")
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=C['Cluster'], palette='Set2', s=100, ax=ax3)
    ax3.set_title(f"Customer Segmentation (K={best_k}) — Silhouette Score: {best_score:.2f}")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.legend(title='Cluster')
    st.pyplot(fig3)

    # Cluster Profiling
    st.subheader("Cluster Profiles (Mean Values)")
    cluster_summary = C.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(cluster_summary)

    # Heatmap of cluster profile
    st.subheader("Cluster Profile Heatmap")
    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.heatmap(cluster_summary.T, cmap='YlGnBu', annot=True, ax=ax4)
    ax4.set_title("Cluster Profiles (Mean Feature Values)")
    st.pyplot(fig4)
else:
    st.info("📥 Please upload your dataset to get started.")